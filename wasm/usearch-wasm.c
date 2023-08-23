#include <stdlib.h>
#include "usearch-wasm.h"
#include <emscripten/emscripten.h>


__attribute__((weak, export_name("canonical_abi_realloc")))
void *canonical_abi_realloc(
void *ptr,
size_t orig_size,
size_t org_align,
size_t new_size
) {
  void *ret = realloc(ptr, new_size);
  if (!ret)
  abort();
  return ret;
}

__attribute__((weak, export_name("canonical_abi_free")))
void canonical_abi_free(
void *ptr,
size_t size,
size_t align
) {
  free(ptr);
}
#include <string.h>

void usearch_wasm_string_set(usearch_wasm_string_t *ret, const char *s) {
  ret->ptr = (char*) s;
  ret->len = strlen(s);
}

void usearch_wasm_string_dup(usearch_wasm_string_t *ret, const char *s) {
  ret->len = strlen(s);
  ret->ptr = canonical_abi_realloc(NULL, 0, 1, ret->len);
  memcpy(ret->ptr, s, ret->len);
}

void usearch_wasm_string_free(usearch_wasm_string_t *ret) {
  canonical_abi_free(ret->ptr, ret->len, 1);
  ret->ptr = NULL;
  ret->len = 0;
}
void usearch_wasm_error_free(usearch_wasm_error_t *ptr) {
  usearch_wasm_string_free(ptr);
}
void usearch_wasm_list_float32_free(usearch_wasm_list_float32_t *ptr) {
  canonical_abi_free(ptr->ptr, ptr->len * 4, 4);
}
void usearch_wasm_list_float64_free(usearch_wasm_list_float64_t *ptr) {
  canonical_abi_free(ptr->ptr, ptr->len * 8, 8);
}
void usearch_wasm_list_s8_free(usearch_wasm_list_s8_t *ptr) {
  canonical_abi_free(ptr->ptr, ptr->len * 1, 1);
}
void usearch_wasm_list_u8_free(usearch_wasm_list_u8_t *ptr) {
  canonical_abi_free(ptr->ptr, ptr->len * 1, 1);
}
void usearch_wasm_vector_free(usearch_wasm_vector_t *ptr) {
  switch ((int32_t) ptr->tag) {
    case 0: {
      usearch_wasm_list_float32_free(&ptr->val.floats32);
      break;
    }
    case 1: {
      usearch_wasm_list_float64_free(&ptr->val.floats64);
      break;
    }
    case 2: {
      usearch_wasm_list_s8_free(&ptr->val.ints8);
      break;
    }
    case 3: {
      usearch_wasm_list_u8_free(&ptr->val.bytes);
      break;
    }
  }
}
__attribute__((export_name("init")))
EMSCRIPTEN_KEEPALIVE int64_t init(int32_t arg, int32_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int32_t arg5, int32_t arg6) {
  usearch_wasm_init_options_t arg7 = (usearch_wasm_init_options_t) {
    arg,
    arg0,
    (uint64_t) (arg1),
    (uint64_t) (arg2),
    (uint64_t) (arg3),
    (uint64_t) (arg4),
  };
  usearch_wasm_error_t arg8 = (usearch_wasm_string_t) { (char*)(arg5), (size_t)(arg6) };
  usearch_wasm_index_t ret = usearch_wasm_init(&arg7, &arg8);
  return (int64_t) (ret);
}
__attribute__((export_name("release")))
EMSCRIPTEN_KEEPALIVE void release(int64_t arg, int32_t arg0, int32_t arg1) {
  usearch_wasm_error_t arg2 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_release((uint64_t) (arg), &arg2);
}
__attribute__((export_name("save")))
EMSCRIPTEN_KEEPALIVE void save(int64_t arg, int32_t arg0, int32_t arg1, int32_t arg2, int32_t arg3) {
  usearch_wasm_string_t arg4 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_error_t arg5 = (usearch_wasm_string_t) { (char*)(arg2), (size_t)(arg3) };
  usearch_wasm_save((uint64_t) (arg), &arg4, &arg5);
}
__attribute__((export_name("load")))
EMSCRIPTEN_KEEPALIVE void load(int64_t arg, int32_t arg0, int32_t arg1, int32_t arg2, int32_t arg3) {
  usearch_wasm_string_t arg4 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_error_t arg5 = (usearch_wasm_string_t) { (char*)(arg2), (size_t)(arg3) };
  usearch_wasm_load((uint64_t) (arg), &arg4, &arg5);
}
__attribute__((export_name("view")))
EMSCRIPTEN_KEEPALIVE void view(int64_t arg, int32_t arg0, int32_t arg1, int32_t arg2, int32_t arg3) {
  usearch_wasm_string_t arg4 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_error_t arg5 = (usearch_wasm_string_t) { (char*)(arg2), (size_t)(arg3) };
  usearch_wasm_view((uint64_t) (arg), &arg4, &arg5);
}
__attribute__((export_name("size")))
EMSCRIPTEN_KEEPALIVE int64_t size(int64_t arg, int32_t arg0, int32_t arg1) {
  usearch_wasm_error_t arg2 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_size_t ret = usearch_wasm_size((uint64_t) (arg), &arg2);
  return (int64_t) (ret);
}
__attribute__((export_name("capacity")))
EMSCRIPTEN_KEEPALIVE int64_t capacity(int64_t arg, int32_t arg0, int32_t arg1) {
  usearch_wasm_error_t arg2 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_size_t ret = usearch_wasm_capacity((uint64_t) (arg), &arg2);
  return (int64_t) (ret);
}
__attribute__((export_name("dimensions")))
EMSCRIPTEN_KEEPALIVE int64_t dimensions(int64_t arg, int32_t arg0, int32_t arg1) {
  usearch_wasm_error_t arg2 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_size_t ret = usearch_wasm_dimensions((uint64_t) (arg), &arg2);
  return (int64_t) (ret);
}
__attribute__((export_name("connectivity")))
EMSCRIPTEN_KEEPALIVE int64_t connectivity(int64_t arg, int32_t arg0, int32_t arg1) {
  usearch_wasm_error_t arg2 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_size_t ret = usearch_wasm_connectivity((uint64_t) (arg), &arg2);
  return (int64_t) (ret);
}
__attribute__((export_name("reserve")))
EMSCRIPTEN_KEEPALIVE void reserve(int64_t arg, int64_t arg0, int32_t arg1, int32_t arg2) {
  usearch_wasm_error_t arg3 = (usearch_wasm_string_t) { (char*)(arg1), (size_t)(arg2) };
  usearch_wasm_reserve((uint64_t) (arg), (uint64_t) (arg0), &arg3);
}
__attribute__((export_name("add")))
EMSCRIPTEN_KEEPALIVE void add(int64_t arg, int64_t arg0, int32_t arg1, int32_t arg2, int32_t arg3, int32_t arg4, int32_t arg5, int32_t arg6) {
  usearch_wasm_vector_t variant;
  variant.tag = arg1;
  switch ((int32_t) variant.tag) {
    case 0: {
      variant.val.floats32 = (usearch_wasm_list_float32_t) { (float*)(arg2), (size_t)(arg3) };
      break;
    }
    case 1: {
      variant.val.floats64 = (usearch_wasm_list_float64_t) { (double*)(arg2), (size_t)(arg3) };
      break;
    }
    case 2: {
      variant.val.ints8 = (usearch_wasm_list_s8_t) { (int8_t*)(arg2), (size_t)(arg3) };
      break;
    }
    case 3: {
      variant.val.bytes = (usearch_wasm_list_u8_t) { (uint8_t*)(arg2), (size_t)(arg3) };
      break;
    }
  }
  usearch_wasm_vector_t arg7 = variant;
  usearch_wasm_error_t arg8 = (usearch_wasm_string_t) { (char*)(arg5), (size_t)(arg6) };
  usearch_wasm_add((uint64_t) (arg), (uint64_t) (arg0), &arg7, arg4, &arg8);
}
__attribute__((export_name("contains")))
EMSCRIPTEN_KEEPALIVE int32_t contains(int64_t arg, int64_t arg0, int32_t arg1, int32_t arg2) {
  usearch_wasm_error_t arg3 = (usearch_wasm_string_t) { (char*)(arg1), (size_t)(arg2) };
  bool ret = usearch_wasm_contains((uint64_t) (arg), (uint64_t) (arg0), &arg3);
  return ret;
}
__attribute__((export_name("search")))
EMSCRIPTEN_KEEPALIVE int64_t search(int64_t arg, int32_t arg0, int32_t arg1, int32_t arg2, int32_t arg3, int64_t arg4, int64_t arg5, float arg6, int32_t arg7, int32_t arg8) {
  usearch_wasm_vector_t variant;
  variant.tag = arg0;
  switch ((int32_t) variant.tag) {
    case 0: {
      variant.val.floats32 = (usearch_wasm_list_float32_t) { (float*)(arg1), (size_t)(arg2) };
      break;
    }
    case 1: {
      variant.val.floats64 = (usearch_wasm_list_float64_t) { (double*)(arg1), (size_t)(arg2) };
      break;
    }
    case 2: {
      variant.val.ints8 = (usearch_wasm_list_s8_t) { (int8_t*)(arg1), (size_t)(arg2) };
      break;
    }
    case 3: {
      variant.val.bytes = (usearch_wasm_list_u8_t) { (uint8_t*)(arg1), (size_t)(arg2) };
      break;
    }
  }
  usearch_wasm_vector_t arg9 = variant;
  usearch_wasm_error_t arg10 = (usearch_wasm_string_t) { (char*)(arg7), (size_t)(arg8) };
  usearch_wasm_size_t ret = usearch_wasm_search((uint64_t) (arg), &arg9, arg3, (uint64_t) (arg4), (uint64_t) (arg5), arg6, &arg10);
  return (int64_t) (ret);
}
__attribute__((export_name("get")))
EMSCRIPTEN_KEEPALIVE int32_t get(int64_t arg, int64_t arg0, int32_t arg1, int32_t arg2, int32_t arg3, int32_t arg4, int32_t arg5, int32_t arg6) {
  usearch_wasm_vector_t variant;
  variant.tag = arg1;
  switch ((int32_t) variant.tag) {
    case 0: {
      variant.val.floats32 = (usearch_wasm_list_float32_t) { (float*)(arg2), (size_t)(arg3) };
      break;
    }
    case 1: {
      variant.val.floats64 = (usearch_wasm_list_float64_t) { (double*)(arg2), (size_t)(arg3) };
      break;
    }
    case 2: {
      variant.val.ints8 = (usearch_wasm_list_s8_t) { (int8_t*)(arg2), (size_t)(arg3) };
      break;
    }
    case 3: {
      variant.val.bytes = (usearch_wasm_list_u8_t) { (uint8_t*)(arg2), (size_t)(arg3) };
      break;
    }
  }
  usearch_wasm_vector_t arg7 = variant;
  usearch_wasm_error_t arg8 = (usearch_wasm_string_t) { (char*)(arg5), (size_t)(arg6) };
  bool ret = usearch_wasm_get((uint64_t) (arg), (uint64_t) (arg0), &arg7, arg4, &arg8);
  return ret;
}
__attribute__((export_name("remove")))
EMSCRIPTEN_KEEPALIVE int32_t remove(int64_t arg, int64_t arg0, int32_t arg1, int32_t arg2) {
  usearch_wasm_error_t arg3 = (usearch_wasm_string_t) { (char*)(arg1), (size_t)(arg2) };
  bool ret = usearch_wasm_remove((uint64_t) (arg), (uint64_t) (arg0), &arg3);
  return ret;
}
