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

using metric_t = index_dense_metric_t;
using distance_t = punned_distance_t;
using index_t = punned_small_t;
using add_result_t = typename index_t::add_result_t;
using search_result_t = typename index_t::search_result_t;
using label_t = typename index_t::label_t;
using id_t = typename index_t::id_t;

class Index : public Napi::ObjectWrap<Index> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    Index(Napi::CallbackInfo const& ctx);

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

    std::unique_ptr<index_t> native_;
};

Napi::Object Index::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass( //
        env, "Index",
        {
            InstanceMethod("dimensions", &Index::GetDimensions),
            InstanceMethod("size", &Index::GetSize),
            InstanceMethod("capacity", &Index::GetCapacity),
            InstanceMethod("connectivity", &Index::GetConnectivity),
            InstanceMethod("add", &Index::Add),
            InstanceMethod("search", &Index::Search),
            InstanceMethod("save", &Index::Save),
            InstanceMethod("load", &Index::Load),
            InstanceMethod("view", &Index::View),
        });

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Index", func);
    return exports;
}

Index::Index(Napi::CallbackInfo const& ctx) : Napi::ObjectWrap<Index>(ctx) {
    Napi::Env env = ctx.Env();

    int length = ctx.Length();
    if (length == 0 || length >= 2 || !ctx[0].IsObject()) {
        Napi::TypeError::New(env, "Pass args as named objects: dimensions: uint, capacity: uint, metric: str")
            .ThrowAsJavaScriptException();
        return;
    }

    Napi::Object params = ctx[0].As<Napi::Object>();
    std::size_t dimensions = params.Has("dimensions") ? params.Get("dimensions").As<Napi::Number>().Uint32Value() : 0;

    index_config_t config;
    index_limits_t limits;
    std::size_t expansion_add = default_expansion_add();
    std::size_t expansion_search = default_expansion_search();

    if (params.Has("capacity"))
        limits.members = params.Get("capacity").As<Napi::Number>().Uint32Value();
    if (params.Has("connectivity"))
        config.connectivity = params.Get("connectivity").As<Napi::Number>().Uint32Value();
    if (params.Has("expansion_add"))
        expansion_add = params.Get("expansion_add").As<Napi::Number>().Uint32Value();
    if (params.Has("expansion_search"))
        expansion_search = params.Get("expansion_search").As<Napi::Number>().Uint32Value();

    scalar_kind_t accuracy = scalar_kind_t::f32_k;
    if (params.Has("accuracy")) {
        std::string accuracy_str = params.Get("accuracy").As<Napi::String>().Utf8Value();
        expected_gt<scalar_kind_t> expected = scalar_kind_from_name(accuracy_str.c_str(), accuracy_str.size());
        if (!expected) {
            Napi::TypeError::New(env, expected.error.what()).ThrowAsJavaScriptException();
            return;
        }
        accuracy = *expected;
    }

    // By default we use the Inner Product similarity
    metric_kind_t metric_kind = metric_kind_t::ip_k;
    if (params.Has("metric")) {
        std::string metric_str = params.Get("metric").As<Napi::String>().Utf8Value();
        expected_gt<metric_kind_t> expected = metric_from_name(metric_str.c_str(), metric_str.size());
        if (!expected) {
            Napi::TypeError::New(env, expected.error.what()).ThrowAsJavaScriptException();
            return;
        }
        metric_kind = *expected;
    }

    native_.reset(
        new index_t(index_t::make(dimensions, metric_kind, config, accuracy, expansion_add, expansion_search)));
    native_->reserve(limits);
}

Napi::Value Index::GetDimensions(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), native_->dimensions());
}
Napi::Value Index::GetSize(Napi::CallbackInfo const& ctx) { return Napi::Number::New(ctx.Env(), native_->size()); }
Napi::Value Index::GetConnectivity(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), native_->connectivity());
}
Napi::Value Index::GetCapacity(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), native_->capacity());
}

void Index::Save(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();

    int length = ctx.Length();
    if (length == 0 || !ctx[0].IsString()) {
        Napi::TypeError::New(env, "Function expects a string path argument").ThrowAsJavaScriptException();
        return;
    }

    try {
        std::string path = ctx[0].As<Napi::String>();
        native_->save(path.c_str());
    } catch (...) {
        Napi::TypeError::New(env, "Serialization failed").ThrowAsJavaScriptException();
    }
}

void Index::Load(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();

    int length = ctx.Length();
    if (length == 0 || !ctx[0].IsString()) {
        Napi::TypeError::New(env, "Function expects a string path argument").ThrowAsJavaScriptException();
        return;
    }

    try {
        std::string path = ctx[0].As<Napi::String>();
        native_->load(path.c_str());
    } catch (...) {
        Napi::TypeError::New(env, "Loading failed").ThrowAsJavaScriptException();
    }
}

void Index::View(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();

    int length = ctx.Length();
    if (length == 0 || !ctx[0].IsString()) {
        Napi::TypeError::New(env, "Function expects a string path argument").ThrowAsJavaScriptException();
        return;
    }

    try {
        std::string path = ctx[0].As<Napi::String>();
        native_->view(path.c_str());
    } catch (...) {
        Napi::TypeError::New(env, "Memory-mapping failed").ThrowAsJavaScriptException();
    }
}

void Index::Add(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();
    if (ctx.Length() < 2 || !ctx[0].IsNumber() || !ctx[1].IsTypedArray())
        return Napi::TypeError::New(env, "Expects an integral label and a float vector").ThrowAsJavaScriptException();

    Napi::Number label_js = ctx[0].As<Napi::Number>();
    Napi::Float32Array vector_js = ctx[1].As<Napi::Float32Array>();

    label_t label = label_js.Uint32Value();
    float const* vector = vector_js.Data();
    std::size_t dimensions = static_cast<std::size_t>(vector_js.ElementLength());
    if (dimensions != native_->dimensions())
        return Napi::TypeError::New(env, "Wrong number of dimensions").ThrowAsJavaScriptException();

    if (native_->size() + 1 >= native_->capacity())
        native_->reserve(ceil2(native_->size() + 1));

    try {
        native_->add(label, vector);
    } catch (std::bad_alloc const&) {
        Napi::TypeError::New(env, "Out of memory").ThrowAsJavaScriptException();
    } catch (...) {
        Napi::TypeError::New(env, "Insertion failed").ThrowAsJavaScriptException();
    }
}

Napi::Value Index::Search(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();
    if (ctx.Length() < 2 || !ctx[0].IsTypedArray() || !ctx[1].IsNumber()) {
        Napi::TypeError::New(env, "Expects a float vector and the number of wanted results")
            .ThrowAsJavaScriptException();
        return {};
    }

    Napi::Float32Array vector_js = ctx[0].As<Napi::Float32Array>();
    Napi::Number wanted_js = ctx[1].As<Napi::Number>();

    float const* vector = vector_js.Data();
    std::size_t dimensions = static_cast<std::size_t>(vector_js.ElementLength());
    std::uint32_t wanted = wanted_js.Uint32Value();
    if (dimensions != native_->dimensions()) {
        Napi::TypeError::New(env, "Wrong number of dimensions").ThrowAsJavaScriptException();
        return {};
    }

    Napi::TypedArrayOf<label_t> matches_js = Napi::TypedArrayOf<label_t>::New(env, wanted);
    Napi::Float32Array distances_js = Napi::Float32Array::New(env, wanted);
    try {
        std::size_t count = native_->search(vector, wanted).dump_to(matches_js.Data(), distances_js.Data());
        Napi::Object result_js = Napi::Object::New(env);
        result_js.Set("labels", matches_js);
        result_js.Set("distances", distances_js);
        result_js.Set("count", Napi::Number::New(env, count));
        return result_js;
    } catch (std::bad_alloc const&) {
        Napi::TypeError::New(env, "Out of memory").ThrowAsJavaScriptException();
        return {};
    } catch (...) {
        Napi::TypeError::New(env, "Search failed").ThrowAsJavaScriptException();
        return {};
    }
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) { return Index::Init(env, exports); }

NODE_API_MODULE(usearch, InitAll)
