import sqlite3
import json
import math
import pytest
import numpy as np

import usearch
from usearch.io import load_matrix, save_matrix
from usearch.index import search
from usearch.eval import random_vectors

from usearch.index import Match, Matches, BatchMatches, Index, Indexes


dimensions = [3, 97, 256]
batch_sizes = [1, 77, 100]


def test_sqlite_distances():
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(usearch.sqlite)

    cursor = conn.cursor()

    # Create a table with additional columns for f32 and f16 BLOBs
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_table (
            id INTEGER PRIMARY KEY,
            vector_json JSON,
            vector_f32 BLOB,
            vector_f16 BLOB
        )
    """
    )

    # Generate and insert random vectors
    num_vectors = 3  # Number of vectors to generate
    dim = 4  # Dimension of each vector
    vectors = []

    for i in range(num_vectors):
        # Generate a random 256-dimensional vector
        vector = np.random.rand(dim)
        vectors.append(vector)

        # Convert the vector to f32 and f16
        vector_f32 = np.float32(vector)
        vector_f16 = np.float16(vector)

        # Insert the vector into the database as JSON and as BLOBs
        cursor.execute(
            """
            INSERT INTO vector_table (vector_json, vector_f32, vector_f16) VALUES (?, ?, ?)
        """,
            (json.dumps(vector.tolist()), vector_f32.tobytes(), vector_f16.tobytes()),
        )

    # Commit changes
    conn.commit()

    similarities = """
    SELECT 
        a.id AS id1,
        b.id AS id2,
        distance_cosine_f32(a.vector_json, b.vector_json) AS cosine_similarity_json,
        distance_cosine_f32(a.vector_f32, b.vector_f32) AS cosine_similarity_f32,
        distance_cosine_f16(a.vector_f16, b.vector_f16) AS cosine_similarity_f16
    FROM 
        vector_table AS a,
        vector_table AS b
    WHERE 
        a.id < b.id;
    """
    cursor.execute(similarities)

    for a, b, similarity_json, similarity_f32, similarity_f16 in cursor.fetchall():
        assert math.isclose(similarity_json, similarity_f32, abs_tol=0.1)
        assert math.isclose(similarity_json, similarity_f16, abs_tol=0.1)
