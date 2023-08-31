#include <stdlib.h>
#include <usearch-wasm.h>

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
void usearch_wasm_keys_free(usearch_wasm_keys_t *ptr) {
  canonical_abi_free(ptr->ptr, ptr->len * 8, 8);
}
void usearch_wasm_distances_free(usearch_wasm_distances_t *ptr) {
  canonical_abi_free(ptr->ptr, ptr->len * 4, 4);
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
void usearch_wasm_expected_index_error_free(usearch_wasm_expected_index_error_t *ptr) {
  if (!ptr->is_err) {
  } else {
    usearch_wasm_error_free(&ptr->val.err);
  }
}
void usearch_wasm_expected_size_error_free(usearch_wasm_expected_size_error_t *ptr) {
  if (!ptr->is_err) {
  } else {
    usearch_wasm_error_free(&ptr->val.err);
  }
}
void usearch_wasm_expected_bool_error_free(usearch_wasm_expected_bool_error_t *ptr) {
  if (!ptr->is_err) {
  } else {
    usearch_wasm_error_free(&ptr->val.err);
  }
}

__attribute__((aligned(8)))
static uint8_t RET_AREA[16];
__attribute__((export_name("init")))
int32_t __wasm_export_usearch_wasm_init(int32_t arg, int32_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4) {
  usearch_wasm_init_options_t arg5 = (usearch_wasm_init_options_t) {
    arg,
    arg0,
    (uint64_t) (arg1),
    (uint64_t) (arg2),
    (uint64_t) (arg3),
    (uint64_t) (arg4),
  };
  usearch_wasm_expected_index_error_t ret;
  usearch_wasm_init(&arg5, &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload6 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 12)) = (int32_t) (*payload6).len;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload6).ptr;
    
  } else {
    const usearch_wasm_index_t *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int64_t*)(ptr + 8)) = (int64_t) (*payload);
    
  }
  return ptr;
}
__attribute__((export_name("release")))
int32_t __wasm_export_usearch_wasm_release(int64_t arg) {
  usearch_wasm_error_t ret;
  usearch_wasm_release((uint64_t) (arg), &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  *((int32_t*)(ptr + 4)) = (int32_t) (ret).len;
  *((int32_t*)(ptr + 0)) = (int32_t) (ret).ptr;
  return ptr;
}
__attribute__((export_name("save")))
int32_t __wasm_export_usearch_wasm_save(int64_t arg, int32_t arg0, int32_t arg1) {
  usearch_wasm_string_t arg2 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_error_t ret;
  usearch_wasm_save((uint64_t) (arg), &arg2, &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  *((int32_t*)(ptr + 4)) = (int32_t) (ret).len;
  *((int32_t*)(ptr + 0)) = (int32_t) (ret).ptr;
  return ptr;
}
__attribute__((export_name("load")))
int32_t __wasm_export_usearch_wasm_load(int64_t arg, int32_t arg0, int32_t arg1) {
  usearch_wasm_string_t arg2 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_error_t ret;
  usearch_wasm_load((uint64_t) (arg), &arg2, &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  *((int32_t*)(ptr + 4)) = (int32_t) (ret).len;
  *((int32_t*)(ptr + 0)) = (int32_t) (ret).ptr;
  return ptr;
}
__attribute__((export_name("view")))
int32_t __wasm_export_usearch_wasm_view(int64_t arg, int32_t arg0, int32_t arg1) {
  usearch_wasm_string_t arg2 = (usearch_wasm_string_t) { (char*)(arg0), (size_t)(arg1) };
  usearch_wasm_error_t ret;
  usearch_wasm_view((uint64_t) (arg), &arg2, &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  *((int32_t*)(ptr + 4)) = (int32_t) (ret).len;
  *((int32_t*)(ptr + 0)) = (int32_t) (ret).ptr;
  return ptr;
}
__attribute__((export_name("size")))
int32_t __wasm_export_usearch_wasm_size(int64_t arg) {
  usearch_wasm_expected_size_error_t ret;
  usearch_wasm_size((uint64_t) (arg), &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload0 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 12)) = (int32_t) (*payload0).len;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload0).ptr;
    
  } else {
    const usearch_wasm_size_t *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int64_t*)(ptr + 8)) = (int64_t) (*payload);
    
  }
  return ptr;
}
__attribute__((export_name("capacity")))
int32_t __wasm_export_usearch_wasm_capacity(int64_t arg) {
  usearch_wasm_expected_size_error_t ret;
  usearch_wasm_capacity((uint64_t) (arg), &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload0 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 12)) = (int32_t) (*payload0).len;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload0).ptr;
    
  } else {
    const usearch_wasm_size_t *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int64_t*)(ptr + 8)) = (int64_t) (*payload);
    
  }
  return ptr;
}
__attribute__((export_name("dimensions")))
int32_t __wasm_export_usearch_wasm_dimensions(int64_t arg) {
  usearch_wasm_expected_size_error_t ret;
  usearch_wasm_dimensions((uint64_t) (arg), &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload0 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 12)) = (int32_t) (*payload0).len;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload0).ptr;
    
  } else {
    const usearch_wasm_size_t *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int64_t*)(ptr + 8)) = (int64_t) (*payload);
    
  }
  return ptr;
}
__attribute__((export_name("connectivity")))
int32_t __wasm_export_usearch_wasm_connectivity(int64_t arg) {
  usearch_wasm_expected_size_error_t ret;
  usearch_wasm_connectivity((uint64_t) (arg), &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload0 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 12)) = (int32_t) (*payload0).len;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload0).ptr;
    
  } else {
    const usearch_wasm_size_t *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int64_t*)(ptr + 8)) = (int64_t) (*payload);
    
  }
  return ptr;
}
__attribute__((export_name("reserve")))
int32_t __wasm_export_usearch_wasm_reserve(int64_t arg, int64_t arg0) {
  usearch_wasm_error_t ret;
  usearch_wasm_reserve((uint64_t) (arg), (uint64_t) (arg0), &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  *((int32_t*)(ptr + 4)) = (int32_t) (ret).len;
  *((int32_t*)(ptr + 0)) = (int32_t) (ret).ptr;
  return ptr;
}
__attribute__((export_name("add")))
int32_t __wasm_export_usearch_wasm_add(int64_t arg, int64_t arg0, int32_t arg1, int32_t arg2, int32_t arg3, int32_t arg4) {
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
  usearch_wasm_vector_t arg5 = variant;
  usearch_wasm_error_t ret;
  usearch_wasm_add((uint64_t) (arg), (uint64_t) (arg0), &arg5, arg4, &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  *((int32_t*)(ptr + 4)) = (int32_t) (ret).len;
  *((int32_t*)(ptr + 0)) = (int32_t) (ret).ptr;
  return ptr;
}
__attribute__((export_name("contains")))
int32_t __wasm_export_usearch_wasm_contains(int64_t arg, int64_t arg0) {
  usearch_wasm_expected_bool_error_t ret;
  usearch_wasm_contains((uint64_t) (arg), (uint64_t) (arg0), &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload1 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload1).len;
    *((int32_t*)(ptr + 4)) = (int32_t) (*payload1).ptr;
    
  } else {
    const bool *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int8_t*)(ptr + 4)) = *payload;
    
  }
  return ptr;
}
__attribute__((export_name("search")))
int32_t __wasm_export_usearch_wasm_search(int64_t arg, int32_t arg0, int32_t arg1, int32_t arg2, int32_t arg3, int64_t arg4, int32_t arg5, int32_t arg6, int32_t arg7, int32_t arg8) {
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
  usearch_wasm_keys_t arg10 = (usearch_wasm_keys_t) { (usearch_wasm_key_t*)(arg5), (size_t)(arg6) };
  usearch_wasm_distances_t arg11 = (usearch_wasm_distances_t) { (usearch_wasm_distance_t*)(arg7), (size_t)(arg8) };
  usearch_wasm_expected_size_error_t ret;
  usearch_wasm_search((uint64_t) (arg), &arg9, arg3, (uint64_t) (arg4), &arg10, &arg11, &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload12 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 12)) = (int32_t) (*payload12).len;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload12).ptr;
    
  } else {
    const usearch_wasm_size_t *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int64_t*)(ptr + 8)) = (int64_t) (*payload);
    
  }
  return ptr;
}
__attribute__((export_name("get")))
int32_t __wasm_export_usearch_wasm_get(int64_t arg, int64_t arg0, int32_t arg1, int32_t arg2, int32_t arg3, int32_t arg4) {
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
  usearch_wasm_vector_t arg5 = variant;
  usearch_wasm_expected_bool_error_t ret;
  usearch_wasm_get((uint64_t) (arg), (uint64_t) (arg0), &arg5, arg4, &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload6 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload6).len;
    *((int32_t*)(ptr + 4)) = (int32_t) (*payload6).ptr;
    
  } else {
    const bool *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int8_t*)(ptr + 4)) = *payload;
    
  }
  return ptr;
}
__attribute__((export_name("remove")))
int32_t __wasm_export_usearch_wasm_remove(int64_t arg, int64_t arg0) {
  usearch_wasm_expected_bool_error_t ret;
  usearch_wasm_remove((uint64_t) (arg), (uint64_t) (arg0), &ret);
  int32_t ptr = (int32_t) &RET_AREA;
  
  if ((ret).is_err) {
    const usearch_wasm_error_t *payload1 = &(ret).val.err;
    *((int8_t*)(ptr + 0)) = 1;
    *((int32_t*)(ptr + 8)) = (int32_t) (*payload1).len;
    *((int32_t*)(ptr + 4)) = (int32_t) (*payload1).ptr;
    
  } else {
    const bool *payload = &(ret).val.ok;
    *((int8_t*)(ptr + 0)) = 0;
    *((int8_t*)(ptr + 4)) = *payload;
    
  }
  return ptr;
}
