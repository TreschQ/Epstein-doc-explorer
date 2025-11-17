#!/usr/bin/env node

import express from 'express';
import cors from 'cors';
import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

const app = express();
const PORT = process.env.PORT || 3001;
const DB_PATH = process.env.DB_PATH || 'document_analysis.db';
const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:5173', 'http://localhost:3000'];

// CORS configuration with origin whitelist
app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (same-origin requests, mobile apps, curl)
    if (!origin) return callback(null, true);

    // Allow localhost origins for development
    if (ALLOWED_ORIGINS.includes(origin)) {
      return callback(null, true);
    }

    // Allow Render deployment domains (*.onrender.com)
    if (origin && (origin.includes('.onrender.com') || origin.endsWith('onrender.com'))) {
      return callback(null, true);
    }

    // Log rejected origins for debugging
    console.warn(`CORS blocked origin: ${origin}`);

    // Reject other origins
    callback(new Error('Not allowed by CORS'));
  },
  credentials: true,
  maxAge: 86400
}));

// Request size limits
app.use(express.json({ limit: '10mb' }));

// Simple rate limiting middleware
const requestCounts = new Map<string, { count: number; resetTime: number }>();
const RATE_LIMIT_WINDOW = 15 * 60 * 1000; // 15 minutes
const RATE_LIMIT_MAX_REQUESTS = 1000; // Max requests per window

app.use((req, res, next) => {
  const ip = req.ip || req.socket.remoteAddress || 'unknown';
  const now = Date.now();
  const userData = requestCounts.get(ip);

  if (!userData || now > userData.resetTime) {
    requestCounts.set(ip, { count: 1, resetTime: now + RATE_LIMIT_WINDOW });
    return next();
  }

  if (userData.count >= RATE_LIMIT_MAX_REQUESTS) {
    return res.status(429).json({ error: 'Too many requests, please try again later' });
  }

  userData.count++;
  next();
});

// Initialize database with error handling
let db: Database.Database;
try {
  db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL'); // Enable WAL mode for better concurrency
  console.log(`✓ Database initialized: ${DB_PATH}`);
} catch (error) {
  console.error('Failed to initialize database:', error);
  process.exit(1);
}

// Load tag clusters with error handling
let tagClusters: any[] = [];
try {
  const clustersPath = path.join(process.cwd(), 'tag_clusters.json');
  tagClusters = JSON.parse(fs.readFileSync(clustersPath, 'utf-8'));
  console.log(`✓ Loaded ${tagClusters.length} tag clusters`);
} catch (error) {
  console.error('Failed to load tag clusters:', error);
  tagClusters = [];
}

// Get all actors (nodes) with alias resolution
app.get('/api/actors', (req, res) => {
  try {
    const actors = db.prepare(`
      SELECT DISTINCT
        COALESCE(ea.canonical_name, rt.actor) as name,
        COUNT(*) as connection_count
      FROM rdf_triples rt
      LEFT JOIN entity_aliases ea ON rt.actor = ea.original_name
      GROUP BY COALESCE(ea.canonical_name, rt.actor)
      ORDER BY connection_count DESC
      LIMIT 100
    `).all();
    res.json(actors);
  } catch (error) {
    console.error('Error in /api/actors:', error);
    res.status(500).json({ error: 'An internal error occurred' });
  }
});

// Helper function to validate and sanitize inputs
function validateLimit(limit: any): number {
  const parsed = parseInt(limit);
  if (isNaN(parsed) || parsed < 1) return 500;
  return Math.min(20000, Math.max(1, parsed));
}

function validateClusterIds(clusters: any): number[] {
  if (!clusters) return [];
  return String(clusters)
    .split(',')
    .map(Number)
    .filter(n => !isNaN(n) && n >= 0 && Number.isInteger(n))
    .slice(0, 50); // Limit to 50 clusters max
}

function validateCategories(categories: any): string[] {
  if (!categories) return [];
  return String(categories)
    .split(',')
    .map(c => c.trim())
    .filter(c => c.length > 0 && c.length < 100) // Reasonable category name length
    .slice(0, 50); // Limit to 50 categories max
}

function validateYearRange(yearMin: any, yearMax: any): [number, number] | null {
  if (!yearMin && !yearMax) return null; // No year filter

  const min = parseInt(yearMin);
  const max = parseInt(yearMax);

  if (isNaN(min) || isNaN(max)) return null;
  if (min < 1970 || max > 2025 || min > max) return null;

  return [min, max];
}

function validateKeywords(keywords: any): string[] {
  if (!keywords) return [];
  return String(keywords)
    .split(',')
    .map(k => k.trim().toLowerCase())
    .filter(k => k.length > 0 && k.length < 100) // Reasonable keyword length
    .slice(0, 20); // Limit to 20 keywords max
}

// BM25 scoring function for fuzzy text matching
function calculateBM25Score(text: string, keywords: string[]): number {
  if (!text || keywords.length === 0) return 0;

  const textLower = text.toLowerCase();
  const words = textLower.split(/\s+/);
  const docLength = words.length;
  const avgDocLength = 100; // Approximate average document length

  // BM25 parameters
  const k1 = 1.2; // Term frequency saturation parameter
  const b = 0.75; // Length normalization parameter

  let score = 0;

  keywords.forEach(keyword => {
    // Count term frequency in document
    const tf = words.filter(word => word.includes(keyword)).length;
    if (tf === 0) return;

    // Simplified IDF (inverse document frequency) - assume keyword appears in 10% of docs
    const idf = Math.log(10);

    // BM25 formula
    const numerator = tf * (k1 + 1);
    const denominator = tf + k1 * (1 - b + b * (docLength / avgDocLength));

    score += idf * (numerator / denominator);
  });

  return score;
}

// No longer needed - we use the materialized top_cluster_ids column instead

// Get all relationships (edges) with distance-based pruning
app.get('/api/relationships', (req, res) => {
  try {
    const limit = validateLimit(req.query.limit);
    const clusterIds = validateClusterIds(req.query.clusters);
    const categories = validateCategories(req.query.categories);
    const yearRange = validateYearRange(req.query.yearMin, req.query.yearMax);
    const includeUndated = req.query.includeUndated !== 'false'; // Default to true
    const keywords = validateKeywords(req.query.keywords);
    const EPSTEIN_NAME = 'Jeffrey Epstein';

    // Build set of selected cluster IDs for filtering
    const selectedClusterIds = new Set<number>(clusterIds);
    const selectedCategories = new Set<string>(categories);

    // Build WHERE clause for categories
    let categoryWhere = '';
    let categoryParams: string[] = [];
    if (selectedCategories.size > 0) {
      const placeholders = Array.from(selectedCategories).map(() => '?').join(',');
      categoryWhere = `AND d.category IN (${placeholders})`;
      categoryParams = Array.from(selectedCategories);
    }

    // Build WHERE clause for year range
    let yearWhere = '';
    let yearParams: string[] = [];
    if (yearRange) {
      const [minYear, maxYear] = yearRange;
      if (includeUndated) {
        yearWhere = `AND (rt.timestamp IS NULL OR (CAST(substr(rt.timestamp, 1, 4) AS INTEGER) >= ? AND CAST(substr(rt.timestamp, 1, 4) AS INTEGER) <= ?))`;
      } else {
        yearWhere = `AND (rt.timestamp IS NOT NULL AND CAST(substr(rt.timestamp, 1, 4) AS INTEGER) >= ? AND CAST(substr(rt.timestamp, 1, 4) AS INTEGER) <= ?)`;
      }
      yearParams = [minYear.toString(), maxYear.toString()];
    }

    // Fetch relationships with alias resolution and triple_tags
    // Apply database-level LIMIT to prevent memory exhaustion
    const MAX_DB_LIMIT = 50000; // Maximum rows to fetch from database
    const allRelationships = db.prepare(`
      SELECT
        rt.id,
        rt.doc_id,
        rt.timestamp,
        COALESCE(ea_actor.canonical_name, rt.actor) as actor,
        rt.action,
        COALESCE(ea_target.canonical_name, rt.target) as target,
        rt.location,
        rt.triple_tags,
        rt.top_cluster_ids
      FROM rdf_triples rt
      LEFT JOIN entity_aliases ea_actor ON rt.actor = ea_actor.original_name
      LEFT JOIN entity_aliases ea_target ON rt.target = ea_target.original_name
      LEFT JOIN documents d ON rt.doc_id = d.doc_id
      WHERE (rt.timestamp IS NULL OR rt.timestamp >= '1970-01-01')
      ${categoryWhere}
      ${yearWhere}
      ORDER BY rt.timestamp
      LIMIT ?
    `).all(...categoryParams, ...yearParams, MAX_DB_LIMIT) as Array<{
      id: number;
      doc_id: string;
      timestamp: string | null;
      actor: string;
      action: string;
      target: string;
      location: string | null;
      triple_tags: string | null;
      top_cluster_ids: string | null;
    }>;

    // Filter by tag clusters if specified
    let filteredRelationships = allRelationships.filter(rel => {
      if (selectedClusterIds.size === 0) return true; // No filter

      try {
        // Use the materialized top_cluster_ids column
        const topClusters = rel.top_cluster_ids ? JSON.parse(rel.top_cluster_ids) : [];
        // Include if any of the top 3 clusters are selected
        return topClusters.some((clusterId: number) => selectedClusterIds.has(clusterId));
      } catch {
        return false;
      }
    });

    // Filter by keywords using BM25 fuzzy matching if specified
    if (keywords.length > 0) {
      filteredRelationships = filteredRelationships.filter(rel => {
        // Build searchable text from relationship fields
        const searchText = `${rel.actor} ${rel.action} ${rel.target} ${rel.location || ''}`;
        const score = calculateBM25Score(searchText, keywords);
        // Include relationships with non-zero BM25 score (at least one keyword match)
        return score > 0;
      });
    }

    // Build adjacency list for BFS
    const adjacency = new Map<string, Set<string>>();

    filteredRelationships.forEach(rel => {
      if (!adjacency.has(rel.actor)) adjacency.set(rel.actor, new Set());
      if (!adjacency.has(rel.target)) adjacency.set(rel.target, new Set());
      adjacency.get(rel.actor)!.add(rel.target);
      adjacency.get(rel.target)!.add(rel.actor);
    });

    // BFS to calculate distances from Jeffrey Epstein
    const distances = new Map<string, number>();
    const queue: string[] = [];

    if (adjacency.has(EPSTEIN_NAME)) {
      distances.set(EPSTEIN_NAME, 0);
      queue.push(EPSTEIN_NAME);

      while (queue.length > 0) {
        const current = queue.shift()!;
        const currentDistance = distances.get(current)!;

        const neighbors = adjacency.get(current) || new Set();
        neighbors.forEach(neighbor => {
          if (!distances.has(neighbor)) {
            distances.set(neighbor, currentDistance + 1);
            queue.push(neighbor);
          }
        });
      }
    }

    // Assign distance score to each relationship (minimum distance of the two actors)
    const relationshipsWithDistance = filteredRelationships.map(rel => {
      const actorDistance = distances.get(rel.actor) ?? Infinity;
      const targetDistance = distances.get(rel.target) ?? Infinity;
      const minDistance = Math.min(actorDistance, targetDistance);

      return {
        ...rel,
        _distance: minDistance
      };
    });

    // Sort by distance (closest first) and take top limit
    relationshipsWithDistance.sort((a, b) => a._distance - b._distance);
    const prunedRelationships = relationshipsWithDistance.slice(0, limit);

    // Remove the _distance field before sending
    const relationships = prunedRelationships.map(({ _distance, triple_tags, ...rel }) => ({
      ...rel,
      tags: triple_tags ? JSON.parse(triple_tags) : []
    }));

    // Return both the relationships and metadata
    res.json({
      relationships,
      totalBeforeLimit: filteredRelationships.length,
      totalBeforeFilter: allRelationships.length
    });
  } catch (error) {
    console.error('Error in /api/relationships:', error);
    res.status(500).json({ error: 'An internal error occurred' });
  }
});

// Get relationships for specific actor with alias resolution and cluster filtering
app.get('/api/actor/:name/relationships', (req, res) => {
  try {
    const { name } = req.params;

    // Validate actor name
    if (!name || name.length > 200) {
      return res.status(400).json({ error: 'Invalid actor name' });
    }

    const clusterIds = validateClusterIds(req.query.clusters);
    const categories = validateCategories(req.query.categories);
    const yearRange = validateYearRange(req.query.yearMin, req.query.yearMax);
    const includeUndated = req.query.includeUndated !== 'false'; // Default to true
    const keywords = validateKeywords(req.query.keywords);

    // Build set of selected cluster IDs and categories for filtering
    const selectedClusterIds = new Set<number>(clusterIds);
    const selectedCategories = new Set<string>(categories);

    // Find all aliases for this name (if it's a canonical name)
    // OR find the canonical name if this is an alias
    const aliasQuery = db.prepare(`
      SELECT original_name FROM entity_aliases WHERE canonical_name = ?
      UNION
      SELECT canonical_name FROM entity_aliases WHERE original_name = ?
      UNION
      SELECT ? as name
    `).all(name, name, name);

    const allNames = aliasQuery.map((row: any) => row.original_name || row.canonical_name || row.name);
    const placeholders = allNames.map(() => '?').join(',');

    // First, get the total count WITHOUT any filters (for the "X of Y" display)
    const totalRelationships = db.prepare(`
      SELECT COUNT(*) as count
      FROM rdf_triples rt
      WHERE (rt.actor IN (${placeholders}) OR rt.target IN (${placeholders}))
        AND (rt.timestamp IS NULL OR rt.timestamp >= '1970-01-01')
    `).get(...allNames, ...allNames) as { count: number };

    // Build WHERE clause for categories
    let categoryWhere = '';
    let categoryParams: string[] = [];
    if (selectedCategories.size > 0) {
      const catPlaceholders = Array.from(selectedCategories).map(() => '?').join(',');
      categoryWhere = `AND d.category IN (${catPlaceholders})`;
      categoryParams = Array.from(selectedCategories);
    }

    // Build WHERE clause for year range
    let yearWhere = '';
    let yearParams: string[] = [];
    if (yearRange) {
      const [minYear, maxYear] = yearRange;
      if (includeUndated) {
        yearWhere = `AND (rt.timestamp IS NULL OR (CAST(substr(rt.timestamp, 1, 4) AS INTEGER) >= ? AND CAST(substr(rt.timestamp, 1, 4) AS INTEGER) <= ?))`;
      } else {
        yearWhere = `AND (rt.timestamp IS NOT NULL AND CAST(substr(rt.timestamp, 1, 4) AS INTEGER) >= ? AND CAST(substr(rt.timestamp, 1, 4) AS INTEGER) <= ?)`;
      }
      yearParams = [minYear.toString(), maxYear.toString()];
    }

    const allRelationships = db.prepare(`
      SELECT
        rt.id,
        rt.doc_id,
        rt.timestamp,
        COALESCE(ea_actor.canonical_name, rt.actor) as actor,
        rt.action,
        COALESCE(ea_target.canonical_name, rt.target) as target,
        rt.location,
        rt.triple_tags,
        rt.top_cluster_ids
      FROM rdf_triples rt
      LEFT JOIN entity_aliases ea_actor ON rt.actor = ea_actor.original_name
      LEFT JOIN entity_aliases ea_target ON rt.target = ea_target.original_name
      LEFT JOIN documents d ON rt.doc_id = d.doc_id
      WHERE (rt.actor IN (${placeholders}) OR rt.target IN (${placeholders}))
        AND (rt.timestamp IS NULL OR rt.timestamp >= '1970-01-01')
        ${categoryWhere}
        ${yearWhere}
      ORDER BY rt.timestamp
    `).all(...allNames, ...allNames, ...categoryParams, ...yearParams) as Array<{
      id: number;
      doc_id: string;
      timestamp: string | null;
      actor: string;
      action: string;
      target: string;
      location: string | null;
      triple_tags: string | null;
      top_cluster_ids: string | null;
    }>;

    // Filter by tag clusters if specified
    let filteredRelationships = allRelationships.filter(rel => {
      if (selectedClusterIds.size === 0) return true; // No filter

      try {
        // Use the materialized top_cluster_ids column
        const topClusters = rel.top_cluster_ids ? JSON.parse(rel.top_cluster_ids) : [];
        // Include if any of the top 3 clusters are selected
        return topClusters.some((clusterId: number) => selectedClusterIds.has(clusterId));
      } catch {
        return false;
      }
    });

    // Filter by keywords using BM25 fuzzy matching if specified
    if (keywords.length > 0) {
      filteredRelationships = filteredRelationships.filter(rel => {
        // Build searchable text from relationship fields
        const searchText = `${rel.actor} ${rel.action} ${rel.target} ${rel.location || ''}`;
        const score = calculateBM25Score(searchText, keywords);
        // Include relationships with non-zero BM25 score (at least one keyword match)
        return score > 0;
      });
    }

    const relationships = filteredRelationships.map((rel) => ({
      id: rel.id,
      doc_id: rel.doc_id,
      timestamp: rel.timestamp,
      actor: rel.actor,
      action: rel.action,
      target: rel.target,
      location: rel.location,
      tags: rel.triple_tags ? JSON.parse(rel.triple_tags) : []
    }));

    res.json({
      relationships,
      totalBeforeFilter: totalRelationships.count
    });
  } catch (error) {
    console.error('Error in /api/actor/:name/relationships:', error);
    res.status(500).json({ error: 'An internal error occurred' });
  }
});

// Get statistics with alias resolution
app.get('/api/stats', (req, res) => {
  try {
    const stats = {
      totalDocuments: db.prepare('SELECT COUNT(*) as count FROM documents').get(),
      totalTriples: db.prepare('SELECT COUNT(*) as count FROM rdf_triples').get(),
      totalActors: db.prepare(`
        SELECT COUNT(DISTINCT COALESCE(ea.canonical_name, rt.actor)) as count
        FROM rdf_triples rt
        LEFT JOIN entity_aliases ea ON rt.actor = ea.original_name
      `).get(),
      categories: db.prepare(`
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC
      `).all(),
    };
    res.json(stats);
  } catch (error) {
    console.error('Error in /api/stats:', error);
    res.status(500).json({ error: 'An internal error occurred' });
  }
});

// Search actors with alias resolution
app.get('/api/search', (req, res) => {
  try {
    const query = req.query.q as string;
    if (!query) {
      return res.json([]);
    }

    const results = db.prepare(`
      SELECT DISTINCT
        COALESCE(ea.canonical_name, rt.actor) as name,
        COUNT(*) as connection_count
      FROM rdf_triples rt
      LEFT JOIN entity_aliases ea ON rt.actor = ea.original_name
      WHERE COALESCE(ea.canonical_name, rt.actor) LIKE ?
      GROUP BY COALESCE(ea.canonical_name, rt.actor)
      ORDER BY connection_count DESC
      LIMIT 20
    `).all(`%${query}%`);

    res.json(results);
  } catch (error) {
    console.error('Error in /api/search:', error);
    res.status(500).json({ error: 'An internal error occurred' });
  }
});

// Get document by doc_id
app.get('/api/document/:docId', (req, res) => {
  try {
    const { docId } = req.params;
    const doc = db.prepare(`
      SELECT
        doc_id,
        file_path,
        one_sentence_summary,
        paragraph_summary,
        category,
        date_range_earliest,
        date_range_latest
      FROM documents
      WHERE doc_id = ?
    `).get(docId);

    if (!doc) {
      return res.status(404).json({ error: 'Document not found' });
    }

    res.json(doc);
  } catch (error) {
    console.error('Error in /api/document/:docId:', error);
    res.status(500).json({ error: 'An internal error occurred' });
  }
});

// Get document text from database
app.get('/api/document/:docId/text', (req, res) => {
  try {
    const { docId } = req.params;

    // Validate docId
    if (!docId || docId.length > 100 || /[<>:"|?*]/.test(docId)) {
      return res.status(400).json({ error: 'Invalid document ID' });
    }

    const doc = db.prepare('SELECT full_text FROM documents WHERE doc_id = ?').get(docId) as { full_text: string | null } | undefined;

    if (!doc) {
      return res.status(404).json({ error: 'Document not found' });
    }

    if (!doc.full_text) {
      return res.status(404).json({ error: 'Document text not available' });
    }

    res.json({ text: doc.full_text });
  } catch (error) {
    console.error('Error in /api/document/:docId/text:', error);
    res.status(500).json({ error: 'An internal error occurred' });
  }
});

// Get tag clusters
app.get('/api/tag-clusters', (req, res) => {
  try {
    // Return just the cluster metadata (id, name, exemplars) without all tags
    const clusters = tagClusters.map((cluster: any) => ({
      id: cluster.id,
      name: cluster.name,
      exemplars: cluster.exemplars,
      tagCount: cluster.tags.length
    }));
    res.json(clusters);
  } catch (error) {
    console.error('Error in /api/tag-clusters:', error);
    res.status(500).json({ error: 'An internal error occurred' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

// Serve static frontend files
const frontendPath = path.join(process.cwd(), 'network-ui', 'dist');
if (fs.existsSync(frontendPath)) {
  app.use(express.static(frontendPath));

  // Serve index.html for all non-API routes (SPA support)
  app.use((req, res, next) => {
    // Skip API routes
    if (req.path.startsWith('/api') || req.path.startsWith('/health')) {
      return next();
    }
    // Serve index.html for all other routes (client-side routing)
    res.sendFile(path.join(frontendPath, 'index.html'));
  });

  console.log(`✓ Serving frontend from ${frontendPath}`);
} else {
  console.log(`⚠ Frontend build not found at ${frontendPath}`);
}

const server = app.listen(PORT, () => {
  console.log(`\n🚀 API Server running at http://localhost:${PORT}`);
  console.log(`📊 Network UI will connect to this server\n`);
});

// Graceful shutdown
const gracefulShutdown = (signal: string) => {
  console.log(`\n${signal} received, closing server gracefully...`);
  server.close(() => {
    console.log('HTTP server closed');
    try {
      db.close();
      console.log('Database connection closed');
    } catch (error) {
      console.error('Error closing database:', error);
    }
    process.exit(0);
  });

  // Force shutdown after 10 seconds
  setTimeout(() => {
    console.error('Forced shutdown after timeout');
    process.exit(1);
  }, 10000);
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
