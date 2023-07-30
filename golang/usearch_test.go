package usearch

import (
	"runtime"
	"testing"
)

func Assure(err error) {
	if err != nil {
		panic(err)
	}
}
func TestUsearch(t *testing.T) {
	runtime.LockOSThread()
	dim := 128
	conf := DefaultConfig(dim)
	ind := NewIndex(conf)
	defer ind.Destroy()
	if ind.VecDimension() != dim {
		t.Fatalf("expected dimension %d, got %d", dim, ind.VecDimension())
	}

	if ind.Connectivity() != conf.Connectivity {
		t.Fatalf("expected connectivity %d, got %d", conf.Connectivity, ind.Connectivity())
	}

	if ind.Capacity() != conf.InitCapacity {
		t.Fatalf("expected initial capacity %d, got %d", conf.InitCapacity, ind.Capacity())
	}

	if ind.Len() != 0 {
		t.Fatalf("expected empty initial index, got %d", ind.Len())
	}

	vec := make([]float32, dim)
	vec[0] = 40.0
	vec[1] = 2.0
	Assure(ind.Add(42, vec))
	if ind.Len() != 1 {
		t.Fatalf("expected index size 1, got %d", ind.Len())
	}

	res := ind.Search(vec, 10)
	if len(res) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res))
	}

	if res[0] != 42 {
		t.Fatalf("expected result 42, got %d", res[0])
	}

	vec_far := make([]float32, dim)
	Assure(ind.Add(43, vec_far))
	vec_close := make([]float32, dim)
	vec_close[0] = 40.0
	vec_close[1] = 4.0
	res = ind.Search(vec_close, 1)
	if len(res) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res))
	}
	if res[0] != 42 {
		t.Fatalf("expected closest vector key to be 42, got %d", res[0])
	}

	res = ind.Search(vec_close, 2)
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %d", len(res))
	}
	if !(res[0] == 42 && res[1] == 43) {
		t.Fatalf("expected closest vector keys to be 42, 43, got %d, %d", res[0], res[1])
	}

	Assure(ind.Reserve(102))
	for i := 0; i < 100; i++ {
		vec[i] += 10
		Assure(ind.Add(i, vec))
	}
	res = ind.Search(vec, 100)
	if len(res) != 100 {
		t.Fatalf("expected 100 results, got %d", len(res))
	}
}
