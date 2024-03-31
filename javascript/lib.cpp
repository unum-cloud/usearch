/**
 *  @file javascript.cpp
 *  @author Ash Vardanian
 *  @brief JavaScript bindings for Unum USearch.
 *  @date 2023-04-26
 *
 *  @copyright Copyright (c) 2023
 *
 *  @see NodeJS docs: https://nodejs.org/api/addons.html#hello-world
 *
 */
#include <new> // `std::bad_alloc`

#define NAPI_CPP_EXCEPTIONS
#include <napi.h>
#include <node_api.h>

#include <usearch/index_dense.hpp>

using namespace unum::usearch;
using namespace unum;

class CompiledIndex : public Napi::ObjectWrap<CompiledIndex> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    CompiledIndex(Napi::CallbackInfo const& ctx);

  private:
    Napi::Value GetDimensions(Napi::CallbackInfo const& ctx);
    Napi::Value GetSize(Napi::CallbackInfo const& ctx);
    Napi::Value GetCapacity(Napi::CallbackInfo const& ctx);
    Napi::Value GetConnectivity(Napi::CallbackInfo const& ctx);

    void Save(Napi::CallbackInfo const& ctx);
    void Load(Napi::CallbackInfo const& ctx);
    void View(Napi::CallbackInfo const& ctx);

    void Add(Napi::CallbackInfo const& ctx);
    Napi::Value Search(Napi::CallbackInfo const& ctx);
    Napi::Value Remove(Napi::CallbackInfo const& ctx);
    Napi::Value Contains(Napi::CallbackInfo const& ctx);
    Napi::Value Count(Napi::CallbackInfo const& ctx);

    std::unique_ptr<index_dense_t> native_;
};

Napi::Object CompiledIndex::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass( //
        env, "CompiledIndex",
        {
            InstanceMethod("dimensions", &CompiledIndex::GetDimensions),
            InstanceMethod("size", &CompiledIndex::GetSize),
            InstanceMethod("capacity", &CompiledIndex::GetCapacity),
            InstanceMethod("connectivity", &CompiledIndex::GetConnectivity),
            InstanceMethod("add", &CompiledIndex::Add),
            InstanceMethod("search", &CompiledIndex::Search),
            InstanceMethod("remove", &CompiledIndex::Remove),
            InstanceMethod("contains", &CompiledIndex::Contains),
            InstanceMethod("count", &CompiledIndex::Count),
            InstanceMethod("save", &CompiledIndex::Save),
            InstanceMethod("load", &CompiledIndex::Load),
            InstanceMethod("view", &CompiledIndex::View),
        });

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("CompiledIndex", func);
    return exports;
}

std::size_t napi_argument_to_size(Napi::Value v) {
    return static_cast<std::size_t>(v.As<Napi::Number>().DoubleValue());
}

CompiledIndex::CompiledIndex(Napi::CallbackInfo const& ctx) : Napi::ObjectWrap<CompiledIndex>(ctx) {

    // Directly assign the parameters without checks
    std::size_t dimensions = napi_argument_to_size(ctx[0]);
    metric_kind_t metric_kind = metric_from_name(ctx[1].As<Napi::String>().Utf8Value().c_str());
    scalar_kind_t quantization = scalar_kind_from_name(ctx[2].As<Napi::String>().Utf8Value().c_str());
    std::size_t connectivity = napi_argument_to_size(ctx[3]);
    std::size_t expansion_add = napi_argument_to_size(ctx[4]);
    std::size_t expansion_search = napi_argument_to_size(ctx[5]);
    bool multi = ctx[6].As<Napi::Boolean>().Value();

    metric_punned_t metric(dimensions, metric_kind, quantization);
    if (metric.missing()) {
        Napi::TypeError::New(ctx.Env(), "Failed to initialize the metric!").ThrowAsJavaScriptException();
        return;
    }

    index_dense_config_t config(connectivity, expansion_add, expansion_search);
    config.multi = multi;
    native_.reset(new index_dense_t(index_dense_t::make(metric, config)));
    if (!native_)
        Napi::Error::New(ctx.Env(), "Out of memory!").ThrowAsJavaScriptException();
}

Napi::Value CompiledIndex::GetDimensions(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), static_cast<std::uint64_t>(native_->dimensions()));
}
Napi::Value CompiledIndex::GetConnectivity(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), static_cast<std::uint64_t>(native_->connectivity()));
}
Napi::Value CompiledIndex::GetSize(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), static_cast<std::uint64_t>(native_->size()));
}
Napi::Value CompiledIndex::GetCapacity(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), static_cast<std::uint64_t>(native_->capacity()));
}

void CompiledIndex::Save(Napi::CallbackInfo const& ctx) {
    try {
        std::string path = ctx[0].As<Napi::String>();
        auto result = native_->save(path.c_str());
        if (!result)
            Napi::TypeError::New(ctx.Env(), result.error.release()).ThrowAsJavaScriptException();

    } catch (...) {
        Napi::TypeError::New(ctx.Env(), "Serialization failed").ThrowAsJavaScriptException();
    }
}

void CompiledIndex::Load(Napi::CallbackInfo const& ctx) {
    try {
        std::string path = ctx[0].As<Napi::String>();
        auto result = native_->load(path.c_str());
        if (!result)
            Napi::TypeError::New(ctx.Env(), result.error.release()).ThrowAsJavaScriptException();

    } catch (...) {
        Napi::TypeError::New(ctx.Env(), "Loading failed").ThrowAsJavaScriptException();
    }
}

void CompiledIndex::View(Napi::CallbackInfo const& ctx) {

    try {
        std::string path = ctx[0].As<Napi::String>();
        auto result = native_->view(path.c_str());
        if (!result)
            Napi::TypeError::New(ctx.Env(), result.error.release()).ThrowAsJavaScriptException();

    } catch (...) {
        Napi::TypeError::New(ctx.Env(), "Memory-mapping failed").ThrowAsJavaScriptException();
    }
}

void CompiledIndex::Add(Napi::CallbackInfo const& ctx) {
    // Extract keys and vectors from arguments
    Napi::BigUint64Array keys = ctx[0].As<Napi::BigUint64Array>();
    std::size_t tasks = keys.ElementLength();

    // Ensure there is enough capacity
    if (native_->size() + tasks >= native_->capacity())
        native_->reserve(ceil2(native_->size() + tasks));

    // Create an instance of the executor with the default number of threads
    auto run_parallel = [&](auto vectors) {
        executor_stl_t executor;
        executor.fixed(tasks, [&](std::size_t /*thread_idx*/, std::size_t task_idx) {
            native_->add(static_cast<default_key_t>(keys[task_idx]), vectors + task_idx * native_->dimensions());
        });
    };

    Napi::TypedArray vectors = ctx[1].As<Napi::TypedArray>();
    if (vectors.TypedArrayType() == napi_float32_array) {
        run_parallel(vectors.As<Napi::Float32Array>().Data());
    } else if (vectors.TypedArrayType() == napi_float64_array) {
        run_parallel(vectors.As<Napi::Float64Array>().Data());
    } else if (vectors.TypedArrayType() == napi_int8_array) {
        run_parallel(vectors.As<Napi::Int8Array>().Data());
    } else {
        Napi::TypeError::New(ctx.Env(),
                             "Unsupported TypedArray. Supported types are Float32Array, Float64Array, and Int8Array.")
            .ThrowAsJavaScriptException();
    }
}

Napi::Value CompiledIndex::Search(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();
    Napi::TypedArray queries = ctx[0].As<Napi::TypedArray>();
    std::size_t tasks = queries.ElementLength() / native_->dimensions();
    std::size_t wanted = napi_argument_to_size(ctx[1]);

    auto run_parallel = [&](auto vectors) -> Napi::Value {
        Napi::Array result_js = Napi::Array::New(env, 3);
        Napi::BigUint64Array matches_js = Napi::BigUint64Array::New(env, tasks * wanted);
        Napi::Float32Array distances_js = Napi::Float32Array::New(env, tasks * wanted);
        Napi::BigUint64Array counts_js = Napi::BigUint64Array::New(env, tasks);

        auto matches_data = matches_js.Data();
        auto distances_data = distances_js.Data();
        auto counts_data = counts_js.Data();

        try {
            bool failed = false;
            executor_stl_t executor;
            executor.fixed(tasks, [&](std::size_t /*thread_idx*/, std::size_t task_idx) {
                auto result = native_->search(vectors + task_idx * native_->dimensions(), wanted);
                if (!result) {
                    failed = true;
                    Napi::TypeError::New(env, result.error.release()).ThrowAsJavaScriptException();
                } else {
                    counts_data[task_idx] = result.dump_to(matches_data + task_idx * native_->dimensions(),
                                                           distances_data + task_idx * native_->dimensions());
                }
            });

            if (failed)
                return env.Null();

        } catch (std::bad_alloc const&) {
            Napi::TypeError::New(env, "Out of memory").ThrowAsJavaScriptException();
            return env.Null();

        } catch (...) {
            Napi::TypeError::New(env, "Search failed").ThrowAsJavaScriptException();
            return env.Null();
        }

        result_js.Set(0u, matches_js);
        result_js.Set(1u, distances_js);
        result_js.Set(2u, counts_js);
        return result_js;
    };

    if (queries.TypedArrayType() == napi_float32_array) {
        return run_parallel(queries.As<Napi::Float32Array>().Data());
    } else if (queries.TypedArrayType() == napi_float64_array) {
        return run_parallel(queries.As<Napi::Float64Array>().Data());
    } else if (queries.TypedArrayType() == napi_int8_array) {
        return run_parallel(queries.As<Napi::Int8Array>().Data());
    } else {
        Napi::TypeError::New(env,
                             "Unsupported TypedArray. Supported types are Float32Array, Float64Array, and Int8Array.")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value CompiledIndex::Remove(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();
    Napi::BigUint64Array keys = ctx[0].As<Napi::BigUint64Array>();
    std::size_t length = keys.ElementLength();
    Napi::Array result = Napi::Array::New(env, length);
    for (std::size_t i = 0; i < length; ++i) {
        result[i] = Napi::Number::New(env, native_->remove(static_cast<default_key_t>(keys[i])).completed);
    }
    return result;
}

Napi::Value CompiledIndex::Contains(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();
    Napi::BigUint64Array keys = ctx[0].As<Napi::BigUint64Array>();
    std::size_t length = keys.ElementLength();
    Napi::Array result = Napi::Array::New(env, length);
    for (std::size_t i = 0; i < length; ++i)
        result[i] = Napi::Boolean::New(env, native_->contains(static_cast<default_key_t>(keys[i])));
    return result;
}

Napi::Value CompiledIndex::Count(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();
    Napi::BigUint64Array keys = ctx[0].As<Napi::BigUint64Array>();
    std::size_t length = keys.ElementLength();
    Napi::Array result = Napi::Array::New(env, length);
    for (std::size_t i = 0; i < length; ++i)
        result[i] = Napi::Boolean::New(env, native_->count(static_cast<default_key_t>(keys[i])));
    return result;
}

Napi::Value exactSearch(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();

    // Extracting parameters directly without additional type checks.
    Napi::TypedArray dataset = ctx[0].As<Napi::TypedArray>();
    Napi::ArrayBuffer datasetBuffer = dataset.ArrayBuffer();
    Napi::TypedArray queries = ctx[1].As<Napi::TypedArray>();
    Napi::ArrayBuffer queriesBuffer = queries.ArrayBuffer();
    std::uint64_t dimensions = napi_argument_to_size(ctx[2]);
    std::uint64_t wanted = napi_argument_to_size(ctx[3]);
    metric_kind_t metric_kind = metric_from_name(ctx[4].As<Napi::String>().Utf8Value().c_str());

    scalar_kind_t quantization;
    std::size_t bytes_per_scalar;
    switch (queries.TypedArrayType()) {
    case napi_float64_array: quantization = scalar_kind_t::f64_k, bytes_per_scalar = 8; break;
    case napi_int8_array: quantization = scalar_kind_t::i8_k, bytes_per_scalar = 1; break;
    default: quantization = scalar_kind_t::f32_k, bytes_per_scalar = 4; break;
    }

    metric_punned_t metric(dimensions, metric_kind, quantization);
    if (!metric) {
        Napi::TypeError::New(env, "Failed to initialize the metric!").ThrowAsJavaScriptException();
        return env.Null();
    }

    executor_default_t executor;
    exact_search_t search;

    // Performing the exact search.
    std::size_t dataset_size = dataset.ElementLength() / dimensions;
    std::size_t queries_size = queries.ElementLength() / dimensions;
    auto results = search(                                     //
        reinterpret_cast<byte_t const*>(datasetBuffer.Data()), //
        dataset_size,                                          //
        dimensions * bytes_per_scalar,                         //
        reinterpret_cast<byte_t const*>(queriesBuffer.Data()), //
        queries_size,                                          //
        dimensions * bytes_per_scalar,                         //
        wanted, metric, executor);

    if (!results)
        Napi::TypeError::New(env, "Out of memory").ThrowAsJavaScriptException();

    // Constructing the result object
    Napi::Array result_js = Napi::Array::New(env, 3);
    Napi::BigUint64Array matches_js = Napi::BigUint64Array::New(env, queries_size * wanted);
    Napi::Float32Array distances_js = Napi::Float32Array::New(env, queries_size * wanted);
    Napi::BigUint64Array counts_js = Napi::BigUint64Array::New(env, queries_size);

    auto matches_data = matches_js.Data();
    auto distances_data = distances_js.Data();
    auto counts_data = counts_js.Data();

    // Export into JS buffers
    for (std::size_t task_idx = 0; task_idx != queries_size; ++task_idx) {
        auto result = results.at(task_idx);
        counts_data[task_idx] = wanted;
        for (std::size_t result_idx = 0; result_idx != wanted; ++result_idx) {
            matches_data[task_idx * wanted + result_idx] = result[result_idx].offset;
            distances_data[task_idx * wanted + result_idx] = result[result_idx].distance;
        }
    }

    result_js.Set(0u, matches_js);
    result_js.Set(1u, distances_js);
    result_js.Set(2u, counts_js);
    return result_js;
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    exports.Set("exactSearch", Napi::Function::New(env, exactSearch));
    return CompiledIndex::Init(env, exports);
}

NODE_API_MODULE(usearch, InitAll)
