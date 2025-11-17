#!/usr/bin/env node

import Database from 'better-sqlite3';
import { pipeline } from '@huggingface/transformers';
import fs from 'fs/promises';

const EMBEDDING_DIM = 32; // Must match cluster_tags.ts
const EMBEDDING_MODEL = 'Qwen3-Embedding-0.6B-ONNX-fp16-32d';

interface TagCluster {
  id: number;
  name: string;
  tags: string[];
  exemplars: string[];
}

interface StoredEmbedding {
  tag: string;
  embedding: string;
  model: string;
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

async function main() {
  console.log('🔄 Incremental Tag Assignment to Existing Clusters\n');

  const db = new Database('document_analysis.db');

  // Load existing clusters
  console.log('📂 Loading existing clusters from tag_clusters.json...');
  const clustersJson = await fs.readFile('tag_clusters.json', 'utf-8');
  const existingClusters: TagCluster[] = JSON.parse(clustersJson);
  console.log(`   Found ${existingClusters.length} existing clusters`);

  // Get all tags from existing clusters
  const existingTags = new Set<string>();
  for (const cluster of existingClusters) {
    for (const tag of cluster.tags) {
      existingTags.add(tag);
    }
  }
  console.log(`   Total tags in clusters: ${existingTags.size}`);

  // Get all tags from database
  console.log('\n📊 Analyzing database tags...');
  const triples = db.prepare('SELECT triple_tags FROM rdf_triples WHERE triple_tags IS NOT NULL').all() as { triple_tags: string }[];

  const allDbTags = new Set<string>();
  const tagCounts = new Map<string, number>();

  triples.forEach(({ triple_tags }) => {
    try {
      const tags = JSON.parse(triple_tags) as string[];
      tags.forEach(tag => {
        allDbTags.add(tag);
        tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
      });
    } catch (e) {
      // Skip invalid JSON
    }
  });

  console.log(`   Total tags in database: ${allDbTags.size}`);

  // Find new tags (in database but not in clusters)
  const newTags = Array.from(allDbTags).filter(tag => !existingTags.has(tag));
  console.log(`   New tags to assign: ${newTags.length}`);

  if (newTags.length === 0) {
    console.log('\n✅ No new tags to assign! All tags are already clustered.');
    db.close();
    return;
  }

  console.log(`\n📋 Sample new tags (showing first 10):`);
  newTags.slice(0, 10).forEach((tag, i) => {
    console.log(`   ${i + 1}. ${tag} (${tagCounts.get(tag)} occurrences)`);
  });

  // Load cached embeddings
  console.log(`\n💾 Loading cached embeddings...`);
  const existingEmbeddings = db.prepare(
    'SELECT tag, embedding, model FROM tag_embeddings WHERE model = ?'
  ).all(EMBEDDING_MODEL) as StoredEmbedding[];

  const embeddingCache = new Map<string, number[]>();
  for (const stored of existingEmbeddings) {
    embeddingCache.set(stored.tag, JSON.parse(stored.embedding));
  }
  console.log(`   Found ${embeddingCache.size} cached embeddings`);

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

  // Generate embeddings for new tags
  const tagsNeedingEmbeddings = newTags.filter(tag => !embeddingCache.has(tag));
  console.log(`\n🤖 Generating embeddings for ${tagsNeedingEmbeddings.length} new tags...`);

  if (tagsNeedingEmbeddings.length > 0) {
    const extractor = await pipeline(
      'feature-extraction',
      'onnx-community/Qwen3-Embedding-0.6B-ONNX',
      { dtype: 'fp16' }
    );

    const insertStmt = db.prepare(
      'INSERT OR REPLACE INTO tag_embeddings (tag, embedding, model) VALUES (?, ?, ?)'
    );

    for (let i = 0; i < tagsNeedingEmbeddings.length; i++) {
      if (i % 50 === 0) {
        console.log(`   Progress: ${i}/${tagsNeedingEmbeddings.length}`);
      }

      const tag = tagsNeedingEmbeddings[i];

      // Generate embedding
      const output = await extractor(tag, {
        pooling: 'last_token',
        normalize: false
      });

      // Truncate to EMBEDDING_DIM and normalize
      let embedding = Array.from(output.data).slice(0, EMBEDDING_DIM);
      const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
      embedding = embedding.map(val => val / magnitude);

      // Cache and save
      embeddingCache.set(tag, embedding);
      insertStmt.run(tag, JSON.stringify(embedding), EMBEDDING_MODEL);
    }

    console.log(`   ✅ Generated and cached ${tagsNeedingEmbeddings.length} embeddings`);
  } else {
    console.log(`   ✅ All new tags already have cached embeddings`);
  }

  // Assign each new tag to nearest cluster
  console.log('\n📍 Assigning new tags to clusters...');
  const assignments = new Map<number, string[]>();

  for (const tag of newTags) {
    const embedding = embeddingCache.get(tag);
    if (!embedding) {
      console.warn(`   ⚠ No embedding for tag: ${tag}`);
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
      assignments.get(bestClusterId)!.push(tag);
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
  console.log(`✅ Complete!`);
  console.log(`   New tags assigned: ${totalNewTagsAssigned}`);
  console.log(`   Clusters updated: ${assignments.size}`);
  console.log(`   Cluster IDs: PRESERVED (no breaking changes)`);
  console.log(`${'='.repeat(60)}\n`);

  console.log('⚠ Next step: Run add_top_clusters_column.ts to update triple assignments');

  db.close();
}

main().catch(console.error);
