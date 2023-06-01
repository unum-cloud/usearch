package usearch

import (
	"sync"
	"testing"
)

func TestParallelism(t *testing.T) {
	totalInserts := 3000
	ind := NewIndex(DefaultConfig(128))
	defer ind.Destroy()

	var v0 = make([]float32, 128)
	v0[0] = 4.4
	var wg sync.WaitGroup
	ind.Reserve(totalInserts)

	// second thread
	wg.Add(1)
	go func() {
		defer wg.Done()
		var v1 = make([]float32, 128)
		for l := 0; l < totalInserts/2; l++ {
			ind.Add(10000+l, v1)
		}
	}()

	for l := 0; l < totalInserts/2; l++ {
		ind.Add(l, v0)
	}
	wg.Wait()
	if ind.Len() != totalInserts {
		t.Errorf("wrong length: expected %d, got %d", totalInserts, ind.Len())
	}

	res := ind.Search(v0[:], 10)
	if len(res) != 10 {
		t.Errorf("wrong search result length: expected %d, got %d", 10, len(res))
	}
}
