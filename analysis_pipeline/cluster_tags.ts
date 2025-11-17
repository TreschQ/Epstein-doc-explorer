#!/usr/bin/env node

import Database from 'better-sqlite3';
import { pipeline } from '@huggingface/transformers';
import fs from 'fs/promises';

const EMBEDDING_DIM = 32; // Truncate from 1024 to 32 dimensions
const EMBEDDING_MODEL = 'Qwen3-Embedding-0.6B-ONNX-fp16-32d';
const MAX_ITERATIONS = 100; // Maximum iterations for convergence

interface TagCount {
  tag: string;
  count: number;
}

interface TagCluster {
  id: number;
  name: string;
  tags: string[];
  exemplars: string[]; // Top tags representing this cluster
}

interface StoredEmbedding {
  tag: string;
  embedding: string;
  model: string;
}

interface CliArgs {
  mode: 'incremental' | 'full';
  numClusters: number;
}

/**
 * Parse command-line arguments
 */
function parseArgs(): CliArgs {
  const args = process.argv.slice(2);

  let mode: 'incremental' | 'full' = 'incremental';
  let numClusters = 30;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--full' || args[i] === '-f') {
      mode = 'full';
    } else if (args[i] === '--incremental' || args[i] === '-i') {
      mode = 'incremental';
    } else if (args[i] === '--clusters' || args[i] === '-k') {
      const val = parseInt(args[i + 1]);
      if (!isNaN(val) && val > 0 && val <= 100) {
        numClusters = val;
        i++; // Skip next arg
      } else {
        console.error(`Invalid cluster count: ${args[i + 1]}`);
        process.exit(1);
      }
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log(`
Tag Clustering Script

Usage:
  npx tsx cluster_tags.ts [options]

Options:
  -i, --incremental     Incremental mode: assign new tags to existing clusters (default)
  -f, --full            Full reclustering: create new clusters from scratch
  -k, --clusters N      Number of clusters (default: 30, max: 100)
  -h, --help            Show this help message

Examples:
  npx tsx cluster_tags.ts                    # Incremental mode, 30 clusters
  npx tsx cluster_tags.ts --incremental      # Same as above
  npx tsx cluster_tags.ts --full             # Full reclustering, 30 clusters
  npx tsx cluster_tags.ts -f -k 40           # Full reclustering with 40 clusters
  npx tsx cluster_tags.ts -k 25              # Incremental mode, override to 25 clusters (if full)
      `);
      process.exit(0);
    }
  }

  return { mode, numClusters };
}

/**
 * Calculate cosine similarity between two normalized vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
  }
  return dotProduct;
}

/**
 * Calculate cluster centroid from tag embeddings
 */
function calculateCentroid(embeddings: number[][]): number[] {
  const dim = embeddings[0].length;
  const centroid = new Array(dim).fill(0);

  // Average all embeddings
  for (const embedding of embeddings) {
    for (let i = 0; i < dim; i++) {
      centroid[i] += embedding[i];
    }
  }

  // Normalize
  const magnitude = Math.sqrt(centroid.reduce((sum, val) => sum + val * val, 0));
  return centroid.map(val => val / magnitude);
}

/**
 * K-means clustering with cosine distance
 */
function kmeans(data: number[][], k: number, maxIterations: number): number[] {
  const n = data.length;
  const dim = data[0].length;

  console.log(`   Initializing ${k} centroids using k-means++...`);

  // Initialize centroids using k-means++ for better initial placement
  const centroids: number[][] = [];
  const firstIdx = Math.floor(Math.random() * n);
  centroids.push([...data[firstIdx]]);

  for (let i = 1; i < k; i++) {
    const distances = new Array(n).fill(0);
    let sumDistances = 0;

    // Calculate distance to nearest centroid for each point
    for (let j = 0; j < n; j++) {
      let minDist = Infinity;
      for (const centroid of centroids) {
        const dist = 1 - cosineSimilarity(data[j], centroid);
        minDist = Math.min(minDist, dist);
      }
      distances[j] = minDist * minDist; // Square for weighted probability
      sumDistances += distances[j];
    }

    // Select next centroid with probability proportional to distance
    let random = Math.random() * sumDistances;
    for (let j = 0; j < n; j++) {
      random -= distances[j];
      if (random <= 0) {
        centroids.push([...data[j]]);
        break;
      }
    }
  }

  console.log(`   Running K-means iterations...`);

  let assignments = new Array(n).fill(0);
  let converged = false;
  let iteration = 0;

  while (!converged && iteration < maxIterations) {
    // Assignment step: assign each point to nearest centroid
    const newAssignments = new Array(n);

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestCluster = 0;

      for (let j = 0; j < k; j++) {
        const dist = 1 - cosineSimilarity(data[i], centroids[j]);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = j;
        }
      }

      newAssignments[i] = bestCluster;
    }

    // Check for convergence
    converged = newAssignments.every((val, idx) => val === assignments[idx]);
    assignments = newAssignments;

    if (converged) {
      console.log(`   Converged after ${iteration + 1} iterations`);
      break;
    }

    // Update step: recalculate centroids
    const clusterSums: number[][] = Array(k).fill(0).map(() => Array(dim).fill(0));
    const clusterCounts = Array(k).fill(0);

    for (let i = 0; i < n; i++) {
      const cluster = assignments[i];
      clusterCounts[cluster]++;
      for (let d = 0; d < dim; d++) {
        clusterSums[cluster][d] += data[i][d];
      }
    }

    // Calculate new centroids and normalize
    for (let j = 0; j < k; j++) {
      if (clusterCounts[j] > 0) {
        for (let d = 0; d < dim; d++) {
          centroids[j][d] = clusterSums[j][d] / clusterCounts[j];
        }
        // Normalize centroid
        const magnitude = Math.sqrt(centroids[j].reduce((sum, val) => sum + val * val, 0));
        centroids[j] = centroids[j].map(val => val / magnitude);
      }
    }

    iteration++;
    if (iteration % 10 === 0) {
      console.log(`   Iteration ${iteration}/${maxIterations}`);
    }
  }

  if (!converged) {
    console.log(`   Stopped after ${maxIterations} iterations (not fully converged)`);
  }

  return assignments;
}

function generateClusterName(exemplars: string[]): string {
  // Take the first exemplar and clean it up for display
  const first = exemplars[0];

  // Convert underscore to space and title case
  return first
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Incremental mode: assign new tags to existing clusters
 */
async function runIncrementalMode(
  db: Database.Database,
  allDbTags: TagCount[],
  embeddingCache: Map<string, number[]>
): Promise<void> {
  console.log('\n🔄 INCREMENTAL MODE: Assigning new tags to existing clusters\n');

  // Load existing clusters
  let existingClusters: TagCluster[];
  try {
    const clustersJson = await fs.readFile('tag_clusters.json', 'utf-8');
    existingClusters = JSON.parse(clustersJson);
    console.log(`📂 Loaded ${existingClusters.length} existing clusters`);
  } catch (error) {
    console.error('⚠ No existing tag_clusters.json found. Please run with --full first.');
    process.exit(1);
  }

  // Get all tags from existing clusters
  const existingTags = new Set<string>();
  for (const cluster of existingClusters) {
    for (const tag of cluster.tags) {
      existingTags.add(tag);
    }
  }
  console.log(`   Total tags in clusters: ${existingTags.size}`);

  // Find new tags (in database but not in clusters)
  const newTags = allDbTags.filter(tagCount => !existingTags.has(tagCount.tag));
  console.log(`   New tags to assign: ${newTags.length}`);

  if (newTags.length === 0) {
    console.log('\n✅ No new tags to assign! All tags are already clustered.');
    return;
  }

  console.log(`\n📋 Sample new tags (showing first 10):`);
  newTags.slice(0, 10).forEach((tagCount, i) => {
    console.log(`   ${i + 1}. ${tagCount.tag} (${tagCount.count} occurrences)`);
  });

  // Calculate cluster centroids
  console.log('\n🎯 Calculating cluster centroids...');
  const clusterCentroids = new Map<number, number[]>();

  for (const cluster of existingClusters) {
    const clusterEmbeddings: number[][] = [];

    for (const tag of cluster.tags) {
      const embedding = embeddingCache.get(tag);
      if (embedding) {
        clusterEmbeddings.push(embedding);
      }
    }

    if (clusterEmbeddings.length > 0) {
      const centroid = calculateCentroid(clusterEmbeddings);
      clusterCentroids.set(cluster.id, centroid);
      console.log(`   Cluster ${cluster.id} (${cluster.name}): ${clusterEmbeddings.length} tags`);
    } else {
      console.warn(`   ⚠ Cluster ${cluster.id} has no embeddings!`);
    }
  }

  // Assign each new tag to nearest cluster
  console.log('\n📍 Assigning new tags to clusters...');
  const assignments = new Map<number, string[]>();

  for (const tagCount of newTags) {
    const embedding = embeddingCache.get(tagCount.tag);
    if (!embedding) {
      console.warn(`   ⚠ No embedding for tag: ${tagCount.tag}`);
      continue;
    }

    let bestClusterId = -1;
    let bestSimilarity = -Infinity;

    // Find nearest cluster centroid
    for (const [clusterId, centroid] of clusterCentroids.entries()) {
      const similarity = cosineSimilarity(embedding, centroid);
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestClusterId = clusterId;
      }
    }

    if (bestClusterId !== -1) {
      if (!assignments.has(bestClusterId)) {
        assignments.set(bestClusterId, []);
      }
      assignments.get(bestClusterId)!.push(tagCount.tag);
    }
  }

  // Display assignment summary
  console.log('\n📊 Assignment Summary:');
  for (const cluster of existingClusters) {
    const newTagsInCluster = assignments.get(cluster.id) || [];
    if (newTagsInCluster.length > 0) {
      console.log(`   Cluster ${cluster.id} (${cluster.name}): +${newTagsInCluster.length} tags`);
      console.log(`      Sample: ${newTagsInCluster.slice(0, 3).join(', ')}${newTagsInCluster.length > 3 ? '...' : ''}`);
    }
  }

  // Update clusters with new tags
  console.log('\n💾 Updating tag_clusters.json...');
  const updatedClusters = existingClusters.map(cluster => {
    const newTagsInCluster = assignments.get(cluster.id) || [];
    if (newTagsInCluster.length > 0) {
      return {
        ...cluster,
        tags: [...cluster.tags, ...newTagsInCluster.sort()]
      };
    }
    return cluster;
  });

  // Save updated clusters
  await fs.writeFile(
    'tag_clusters.json',
    JSON.stringify(updatedClusters, null, 2)
  );

  console.log('   ✅ Saved updated clusters');

  // Summary
  const totalNewTagsAssigned = Array.from(assignments.values()).reduce((sum, tags) => sum + tags.length, 0);
  console.log(`\n${'='.repeat(60)}`);
  console.log(`✅ Incremental Update Complete!`);
  console.log(`   New tags assigned: ${totalNewTagsAssigned}`);
  console.log(`   Clusters updated: ${assignments.size}`);
  console.log(`   Cluster IDs: PRESERVED (no breaking changes)`);
  console.log(`${'='.repeat(60)}\n`);

  console.log('⚠ Next step: Run add_top_clusters_column.ts to update triple assignments');
}

/**
 * Full mode: create new clusters from scratch
 */
async function runFullMode(
  db: Database.Database,
  allDbTags: TagCount[],
  embeddingCache: Map<string, number[]>,
  numClusters: number
): Promise<void> {
  console.log(`\n🔄 FULL RECLUSTERING MODE: Creating ${numClusters} clusters from scratch\n`);

  console.log(`📊 Clustering ${allDbTags.length} tags...`);

  // Build embeddings array in same order as allDbTags
  const embeddings: number[][] = allDbTags.map(t => embeddingCache.get(t.tag)!);

  console.log(`\n🎯 Clustering tags using K-means...`);
  console.log(`   k=${numClusters}, max iterations=${MAX_ITERATIONS}`);

  // K-means clustering
  const clusterAssignments = kmeans(embeddings, numClusters, MAX_ITERATIONS);

  console.log(`   Clustering complete!`);

  // Organize clusters
  const tagClusters: TagCluster[] = [];

  for (let clusterId = 0; clusterId < numClusters; clusterId++) {
    const clusterTags = allDbTags
      .map((tagCount, idx) => ({ ...tagCount, idx }))
      .filter(item => clusterAssignments[item.idx] === clusterId)
      .sort((a, b) => b.count - a.count);

    if (clusterTags.length === 0) continue;

    // Get top 5 most frequent tags as exemplars
    const exemplars = clusterTags.slice(0, 5).map(t => t.tag);

    // Generate a cluster name from the top exemplar
    const clusterName = generateClusterName(exemplars);

    tagClusters.push({
      id: clusterId,
      name: clusterName,
      tags: clusterTags.map(t => t.tag),
      exemplars
    });

    console.log(`\nCluster ${clusterId}: "${clusterName}"`);
    console.log(`  Tags: ${clusterTags.length}`);
    console.log(`  Top tags: ${exemplars.join(', ')}`);
  }

  // Save results
  console.log(`\n💾 Preparing to save ${tagClusters.length} clusters...`);
  const jsonString = JSON.stringify(tagClusters, null, 2);
  console.log(`   JSON string size: ${(jsonString.length / 1024 / 1024).toFixed(2)} MB`);

  await fs.writeFile(
    'tag_clusters.json',
    jsonString
  );

  console.log(`✅ Saved ${tagClusters.length} clusters to tag_clusters.json`);

  console.log(`\n${'='.repeat(60)}`);
  console.log(`✅ Full Clustering Complete!`);
  console.log(`   Clusters created: ${tagClusters.length}`);
  console.log(`   Total tags: ${allDbTags.length}`);
  console.log(`${'='.repeat(60)}\n`);

  console.log('⚠ Next step: Run add_top_clusters_column.ts to update triple assignments');
}

async function main() {
  const { mode, numClusters } = parseArgs();

  console.log('🔍 Tag Clustering Pipeline');
  console.log(`   Mode: ${mode.toUpperCase()}`);
  console.log(`   Clusters: ${numClusters}`);
  console.log();

  console.log('📊 Loading tags from database...');
  const db = new Database('document_analysis.db');

  // Get all tags with their frequencies
  const triples = db.prepare('SELECT triple_tags FROM rdf_triples WHERE triple_tags IS NOT NULL').all() as { triple_tags: string }[];

  const tagCounts = new Map<string, number>();

  triples.forEach(({ triple_tags }) => {
    try {
      const tags = JSON.parse(triple_tags) as string[];
      tags.forEach(tag => {
        tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
      });
    } catch (e) {
      // Skip invalid JSON
    }
  });

  // Use ALL tags, sorted by frequency
  const allTags: TagCount[] = Array.from(tagCounts.entries())
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count);

  console.log(`   Found ${allTags.length} unique tags`);

  // Load existing embeddings from database
  console.log(`\n💾 Checking for cached embeddings...`);
  const existingEmbeddings = db.prepare(
    'SELECT tag, embedding, model FROM tag_embeddings WHERE model = ?'
  ).all(EMBEDDING_MODEL) as StoredEmbedding[];

  const embeddingCache = new Map<string, number[]>();
  for (const stored of existingEmbeddings) {
    embeddingCache.set(stored.tag, JSON.parse(stored.embedding));
  }
  console.log(`   Found ${embeddingCache.size} cached embeddings`);

  // Determine which tags need embedding generation
  const tagsNeedingEmbeddings = allTags.filter(t => !embeddingCache.has(t.tag));
  console.log(`   Need to generate ${tagsNeedingEmbeddings.length} new embeddings`);

  // Generate missing embeddings
  if (tagsNeedingEmbeddings.length > 0) {
    console.log(`\n🤖 Loading Qwen3-Embedding-0.6B-ONNX model with fp16...`);
    const extractor = await pipeline(
      'feature-extraction',
      'onnx-community/Qwen3-Embedding-0.6B-ONNX',
      { dtype: 'fp16' }
    );

    console.log(`📝 Generating ${EMBEDDING_DIM}d embeddings for ${tagsNeedingEmbeddings.length} tags...`);

    const insertStmt = db.prepare(
      'INSERT OR REPLACE INTO tag_embeddings (tag, embedding, model) VALUES (?, ?, ?)'
    );

    const insertMany = db.transaction((tags: TagCount[]) => {
      for (const tagCount of tags) {
        const embedding = embeddingCache.get(tagCount.tag)!;
        insertStmt.run(tagCount.tag, JSON.stringify(embedding), EMBEDDING_MODEL);
      }
    });

    const newEmbeddings: TagCount[] = [];

    for (let i = 0; i < tagsNeedingEmbeddings.length; i++) {
      if (i % 100 === 0) {
        console.log(`  Progress: ${i}/${tagsNeedingEmbeddings.length}`);
      }

      const tagCount = tagsNeedingEmbeddings[i];

      // Generate full embedding without normalization
      const output = await extractor(tagCount.tag, {
        pooling: 'last_token',
        normalize: false
      });

      // Truncate to first EMBEDDING_DIM dimensions
      let embedding = Array.from(output.data).slice(0, EMBEDDING_DIM);

      // Normalize the truncated embedding
      const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
      embedding = embedding.map(val => val / magnitude);

      embeddingCache.set(tagCount.tag, embedding);
      newEmbeddings.push(tagCount);

      // Batch insert every 100 embeddings
      if (newEmbeddings.length >= 100) {
        insertMany(newEmbeddings);
        newEmbeddings.length = 0;
      }
    }

    // Insert remaining embeddings
    if (newEmbeddings.length > 0) {
      insertMany(newEmbeddings);
    }

    console.log(`✅ Saved ${tagsNeedingEmbeddings.length} new embeddings to database`);
  } else {
    console.log(`✅ All embeddings loaded from cache!`);
  }

  // Run appropriate mode
  if (mode === 'incremental') {
    await runIncrementalMode(db, allTags, embeddingCache);
  } else {
    await runFullMode(db, allTags, embeddingCache, numClusters);
  }

  db.close();
}

main().catch(console.error);
