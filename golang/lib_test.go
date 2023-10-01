package usearch

import (
	"runtime"
	"testing"
	"math"
)

func TestUSearch(t *testing.T) {
	runtime.LockOSThread()

	t.Run("Test Index Initialization", func(t *testing.T) {
		dim := uint(128)
		conf := DefaultConfig(dim)
		ind, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer ind.Destroy()

		found_dims, err := ind.Dimensions()
		if err != nil {
			t.Fatalf("Failed to retrieve dimensions: %s", err)
		}
		if found_dims != dim {
			t.Fatalf("Expected %d dimensions, got %d", dim, found_dims)
		}

		found_len, err := ind.Len()
		if err != nil {
			t.Fatalf("Failed to retrieve size: %s", err)
		}
		if found_len != 0 {
			t.Fatalf("Expected size to be 0, got %d", found_len)
		}

		err = ind.Reserve(100)
		if err != nil {
			t.Fatalf("Failed to reserve capacity: %s", err)
		}
	})

	t.Run("Test Insertion", func(t *testing.T) {
		dim := uint(128)
		conf := DefaultConfig(dim)
		ind, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer ind.Destroy()

		err = ind.Reserve(100)
		if err != nil {
			t.Fatalf("Failed to reserve capacity: %s", err)
		}

		vec := make([]float32, dim)
		vec[0] = 40.0
		vec[1] = 2.0

		err = ind.Add(42, vec)
		if err != nil {
			t.Fatalf("Failed to insert: %s", err)
		}

		found_len, err := ind.Len()
		if err != nil {
			t.Fatalf("Failed to retrieve size after insertion: %s", err)
		}
		if found_len != 1 {
			t.Fatalf("Expected size to be 1, got %d", found_len)
		}
	})

	t.Run("Test Search", func(t *testing.T) {
		dim := uint(128)
		conf := DefaultConfig(dim)
		ind, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer ind.Destroy()

		err = ind.Reserve(100)
		if err != nil {
			t.Fatalf("Failed to reserve capacity: %s", err)
		}

		vec := make([]float32, dim)
		vec[0] = 40.0
		vec[1] = 2.0

		err = ind.Add(42, vec)
		if err != nil {
			t.Fatalf("Failed to insert: %s", err)
		}

		keys, distances, err := ind.Search(vec, 10)
		if err != nil {
			t.Fatalf("Failed to search: %s", err)
		}

		const tolerance = 1e-2  // For example, this sets the tolerance to 0.01
		if keys[0] != 42 || math.Abs(float64(distances[0])) > tolerance {
			t.Fatalf("Expected result 42 with distance 0, got key %d with distance %f", keys[0], distances[0])
		}
	})
}
