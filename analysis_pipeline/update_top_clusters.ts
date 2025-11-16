#!/usr/bin/env node

import Database from 'better-sqlite3';
import fs from 'fs/promises';

interface TagCluster {
  id: number;
  name: string;
  tags: string[];
  exemplars: string[];
}

async function main() {
  console.log('📊 Updating top_cluster_ids column in database...');

  const db = new Database('document_analysis.db');

  // Load tag clusters
  const clustersJson = await fs.readFile('tag_clusters.json', 'utf-8');
  const tagClusters = JSON.parse(clustersJson) as TagCluster[];

  console.log(`✓ Loaded ${tagClusters.length} clusters`);

  // Build a map of tag -> cluster ID
  const tagToClusterMap = new Map<string, number>();
  for (const cluster of tagClusters) {
    for (const tag of cluster.tags) {
      tagToClusterMap.set(tag, cluster.id);
    }
  }

  console.log(`✓ Mapped ${tagToClusterMap.size} tags to clusters`);

  // Get all triples with their tags
  const triples = db.prepare(`
    SELECT id, triple_tags
    FROM rdf_triples
    WHERE triple_tags IS NOT NULL
  `).all() as Array<{ id: number; triple_tags: string }>;

  console.log(`✓ Found ${triples.length} triples to update`);

  // Prepare update statement
  const updateStmt = db.prepare(`
    UPDATE rdf_triples
    SET top_cluster_ids = ?
    WHERE id = ?
  `);

  // Process in transaction for speed
  const updateMany = db.transaction((triples: Array<{ id: number; triple_tags: string }>) => {
    let updated = 0;
    let noMatch = 0;

    for (const triple of triples) {
      try {
        const tags = JSON.parse(triple.triple_tags) as string[];

        // Count how many tags match each cluster
        const clusterCounts = new Map<number, number>();
        for (const tag of tags) {
          const clusterId = tagToClusterMap.get(tag);
          if (clusterId !== undefined) {
            clusterCounts.set(clusterId, (clusterCounts.get(clusterId) || 0) + 1);
          }
        }

        if (clusterCounts.size === 0) {
          // No matching clusters - could assign to "Misc" cluster (20) if it exists
          updateStmt.run(JSON.stringify([20]), triple.id);
          noMatch++;
        } else {
          // Get top 3 clusters by frequency
          const sortedClusters = Array.from(clusterCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([clusterId]) => clusterId);

          updateStmt.run(JSON.stringify(sortedClusters), triple.id);
          updated++;
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }

    console.log(`✓ Updated ${updated} triples with cluster matches`);
    console.log(`✓ Assigned ${noMatch} triples to Misc cluster (no tag matches)`);
  });

  updateMany(triples);

  console.log('\n✅ Database update complete!');

  db.close();
}

main().catch(console.error);
