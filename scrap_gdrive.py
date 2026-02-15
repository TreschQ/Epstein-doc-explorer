import asyncio
import aiohttp
import aiofiles
import boto3
import os
import pickle
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor

import dotenv
dotenv.load_dotenv()

# --- Config ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
LOCAL_DIR = Path("./tmp_download")
GDRIVE_FOLDER_ID = "1Z9PzcddcLnBT2Nw6w5EJNYubxttyuxP-"
MAX_CONCURRENT_DOWNLOADS = 10
MAX_CONCURRENT_UPLOADS = 5

# --- S3 Config ---
s3 = boto3.client("s3",
    endpoint_url="https://t3.storageapi.dev",
    aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("BUCKET_REGION", "us-west-1"),
)
BUCKET_NAME = os.environ["BUCKET_NAME"]


class CredentialManager:
    """Manages Google Drive credentials with auto-refresh."""

    def __init__(self):
        self.creds = None
        self._lock = asyncio.Lock()
        self._load_credentials()

    def _load_credentials(self):
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as f:
                self.creds = pickle.load(f)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                self.creds = flow.run_local_server(port=0)

            with open('token.pickle', 'wb') as f:
                pickle.dump(self.creds, f)

    async def get_valid_token(self):
        """Get a valid access token, refreshing if needed."""
        async with self._lock:
            if not self.creds.valid:
                if self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                    with open('token.pickle', 'wb') as f:
                        pickle.dump(self.creds, f)
                    print("🔄 Token refreshed")
            return self.creds.token


def list_existing_s3_keys():
    """List all existing keys in S3 bucket under epstein/ prefix."""
    existing = set()
    paginator = s3.get_paginator('list_objects_v2')

    print("📦 Checking existing files in S3...")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="epstein/"):
        for obj in page.get('Contents', []):
            # Remove 'epstein/' prefix to match gdrive paths
            key = obj['Key'].replace("epstein/", "", 1)
            existing.add(key)

    print(f"📦 Found {len(existing)} files already in S3\n")
    return existing


def list_files_recursive(service, folder_id, path=""):
    """Recursively list all files in a folder."""
    files = []
    query = f"'{folder_id}' in parents and trashed = false"

    page_token = None
    while True:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token
        ).execute()

        for item in response.get('files', []):
            item_path = f"{path}/{item['name']}" if path else item['name']

            if item['mimeType'] == 'application/vnd.google-apps.folder':
                print(f"📁 Folder: {item_path}")
                files.extend(list_files_recursive(service, item['id'], item_path))
            else:
                files.append({
                    'id': item['id'],
                    'name': item['name'],
                    'path': item_path
                })

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return files


def upload_single_file(filepath: Path):
    """Upload a single file to S3 (blocking, runs in thread)."""
    key = f"epstein/{filepath.relative_to(LOCAL_DIR)}"
    content_type = "image/jpeg" if filepath.suffix.lower() in ['.jpg', '.jpeg'] else "application/octet-stream"

    size_kb = filepath.stat().st_size / 1024

    s3.upload_file(
        str(filepath),
        BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": content_type}
    )

    filepath.unlink()  # Delete immediately after upload
    print(f"✅ S3: {key} ({size_kb:.0f} KB)")


async def download_worker(
    session: aiohttp.ClientSession,
    cred_manager: CredentialManager,
    download_queue: asyncio.Queue,
    upload_queue: asyncio.Queue,
    semaphore: asyncio.Semaphore
):
    """Worker that downloads files and puts them in upload queue."""
    while True:
        file_info = await download_queue.get()
        if file_info is None:  # Poison pill
            download_queue.task_done()
            break

        file_id = file_info['id']
        file_path = LOCAL_DIR / file_info['path']

        # Skip if already exists
        if file_path.exists():
            print(f"⏭️  Skip: {file_info['path']}")
            await upload_queue.put(file_path)
            download_queue.task_done()
            continue

        file_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        access_token = await cred_manager.get_valid_token()
        headers = {"Authorization": f"Bearer {access_token}"}

        async with semaphore:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        async with aiofiles.open(file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        print(f"📥 Downloaded: {file_info['path']}")
                        await upload_queue.put(file_path)
                    else:
                        print(f"❌ Download failed ({response.status}): {file_info['path']}")
            except Exception as e:
                print(f"❌ Error: {file_info['path']} - {e}")

        download_queue.task_done()


async def upload_worker(
    upload_queue: asyncio.Queue,
    executor: ThreadPoolExecutor,
    loop: asyncio.AbstractEventLoop
):
    """Worker that uploads files to S3 as they arrive."""
    while True:
        filepath = await upload_queue.get()
        if filepath is None:  # Poison pill
            upload_queue.task_done()
            break

        try:
            await loop.run_in_executor(executor, upload_single_file, filepath)
        except Exception as e:
            print(f"❌ Upload error: {filepath} - {e}")

        upload_queue.task_done()


async def main():
    # 1. Check existing S3 files
    existing_keys = list_existing_s3_keys()

    # 2. Auth
    print("🔐 Authenticating with Google Drive...")
    cred_manager = CredentialManager()

    # 3. List files from GDrive
    print("📋 Listing files from Google Drive...")
    service = build('drive', 'v3', credentials=cred_manager.creds)
    all_files = list_files_recursive(service, GDRIVE_FOLDER_ID)
    print(f"\n📊 Found {len(all_files)} files in Google Drive")

    # 4. Filter out already uploaded files
    files = [f for f in all_files if f['path'] not in existing_keys]
    skipped = len(all_files) - len(files)
    print(f"⏭️  Skipping {skipped} files already in S3")
    print(f"📥 {len(files)} files to download\n")

    if not files:
        print("✅ All files already uploaded!")
        return

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Setup queues and workers
    download_queue = asyncio.Queue()
    upload_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    loop = asyncio.get_event_loop()

    # Add files to download queue
    for f in files:
        await download_queue.put(f)

    # Create workers
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession(connector=connector) as session:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_UPLOADS) as executor:

            # Start download workers
            download_workers = [
                asyncio.create_task(
                    download_worker(session, cred_manager, download_queue, upload_queue, semaphore)
                )
                for _ in range(MAX_CONCURRENT_DOWNLOADS)
            ]

            # Start upload workers
            upload_workers = [
                asyncio.create_task(
                    upload_worker(upload_queue, executor, loop)
                )
                for _ in range(MAX_CONCURRENT_UPLOADS)
            ]

            # Wait for all downloads to complete
            await download_queue.join()

            # Send poison pills to download workers
            for _ in download_workers:
                await download_queue.put(None)
            await asyncio.gather(*download_workers)

            # Wait for all uploads to complete
            await upload_queue.join()

            # Send poison pills to upload workers
            for _ in upload_workers:
                await upload_queue.put(None)
            await asyncio.gather(*upload_workers)

    print(f"\n🎉 Done! All files uploaded to {BUCKET_NAME}")


if __name__ == "__main__":
    asyncio.run(main())
