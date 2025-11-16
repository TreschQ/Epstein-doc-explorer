#!/usr/bin/env node

import Database from 'better-sqlite3';
import { pipeline } from '@huggingface/transformers';
import fs from 'fs/promises';

const EMBEDDING_DIM = 32; // Truncate from 1024 to 32 dimensions
const EMBEDDING_MODEL = 'Qwen3-Embedding-0.6B-ONNX-fp16-32d';

// K-means parameters
const NUM_CLUSTERS = 30; // Number of clusters
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

async function main() {
  console.log('🔍 Loading tags from database...');
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

  console.log(`📊 Found ${allTags.length} unique tags`);

  // Use ALL tags
  const tagsToCluster = allTags;

  console.log(`📊 Clustering ${tagsToCluster.length} tags...`);

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
  const tagsNeedingEmbeddings = tagsToCluster.filter(t => !embeddingCache.has(t.tag));
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

  // Build embeddings array in same order as tagsToCluster
  const embeddings: number[][] = tagsToCluster.map(t => embeddingCache.get(t.tag)!);

  console.log(`\n🎯 Clustering tags using K-means...`);
  console.log(`   k=${NUM_CLUSTERS}, max iterations=${MAX_ITERATIONS}`);

  // K-means clustering
  const clusterAssignments = kmeans(embeddings, NUM_CLUSTERS, MAX_ITERATIONS);

  console.log(`   Clustering complete!`);

  // Organize clusters
  const tagClusters: TagCluster[] = [];

  for (let clusterId = 0; clusterId < NUM_CLUSTERS; clusterId++) {
    const clusterTags = tagsToCluster
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

  db.close();
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

/**
 * Cosine similarity for normalized vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
  }
  return dotProduct; // Already normalized, so no need to divide by magnitudes
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

main().catch(console.error);
