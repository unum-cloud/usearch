#import "ObjectiveUSearch.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#import "advanced.hpp"
#pragma clang diagnostic pop

using namespace unum::usearch;
using namespace unum;

using distance_t = punned_distance_t;
using native_index_t = auto_index_gt<UInt32>;
using shared_index_t = std::shared_ptr<native_index_t>;

@interface Index ()

@property (readonly) shared_index_t native;

- (instancetype)initWithIndex:(shared_index_t)native;

@end

@implementation Index

- (instancetype)initWithIndex:(shared_index_t)native {
    self = [super init];
    _native = native;
    return self;
}

- (Boolean)isEmpty {
    return _native->size() != 0;
}

- (UInt)dimensions {
    return static_cast<UInt>(_native->dimensions());
}

- (UInt)connectivity {
    return static_cast<UInt>(_native->connectivity());
}

- (UInt)length {
    return static_cast<UInt>(_native->size());
}

- (UInt)capacity {
    return static_cast<UInt>(_native->capacity());
}

- (UInt)expansion_add {
    return static_cast<UInt>(_native->config().expansion_add);
}

- (UInt)expansion_search {
    return static_cast<UInt>(_native->config().expansion_search);
}

+ (instancetype)indexIP:(UInt)dimensions connectivity:(UInt)connectivity {
    std::size_t dims = static_cast<std::size_t>(dimensions);
    config_t config;
    config.connectivity = static_cast<std::size_t>(connectivity);
    shared_index_t ptr = std::make_shared<native_index_t>(native_index_t::ip(dims, accuracy_t::f32_k, config));
    return [[Index alloc] initWithIndex:ptr];
}

+ (instancetype)indexL2:(UInt)dimensions connectivity:(UInt)connectivity {
    std::size_t dims = static_cast<std::size_t>(dimensions);
    config_t config;
    config.connectivity = static_cast<std::size_t>(connectivity);
    shared_index_t ptr = std::make_shared<native_index_t>(native_index_t::l2(dims, accuracy_t::f32_k, config));
    return [[Index alloc] initWithIndex:ptr];
}

+ (instancetype)indexHaversine:(UInt)connectivity {
    config_t config;
    config.connectivity = static_cast<std::size_t>(connectivity);
    shared_index_t ptr = std::make_shared<native_index_t>(native_index_t::haversine(accuracy_t::f32_k, config));
    return [[Index alloc] initWithIndex:ptr];
}

- (void)addSingle:(UInt32)label
           vector:(Float32 const * _Nonnull)vector {
    _native->add(label, vector);
}

- (UInt)searchSingle:(Float32 const* _Nonnull)vector
               count:(UInt)wanted
              labels:(UInt32* _Nullable)labels
           distances:(Float32* _Nullable)distances {
    std::size_t found = _native->search(vector, static_cast<std::size_t>(wanted), labels, distances);
    return static_cast<UInt>(found);
}

- (void)addPrecise:(UInt32)label
            vector:(Float64 const * _Nonnull)vector {
    _native->add(label, (f64_t const *)vector);
}

- (UInt)searchPrecise:(Float64 const* _Nonnull)vector
                count:(UInt)wanted
               labels:(UInt32* _Nullable)labels
            distances:(Float32* _Nullable)distances {
    std::size_t found = _native->search((f64_t const *)vector, static_cast<std::size_t>(wanted), labels, distances);
    return static_cast<UInt>(found);
}

- (void)addImprecise:(UInt32)label
              vector:(void const * _Nonnull)vector {
    _native->add(label, (f16_converted_t const *)vector);
}

- (UInt)searchImprecise:(void const* _Nonnull)vector
                  count:(UInt)wanted
                 labels:(UInt32* _Nullable)labels
              distances:(Float32* _Nullable)distances {
    std::size_t found = _native->search((f16_converted_t const *)vector, static_cast<std::size_t>(wanted), labels, distances);
    return static_cast<UInt>(found);
}

- (void)clear { _native->clear(); }

- (void)save:(NSString*)path {
    char const *path_c = [path UTF8String];
    if (!path_c)
        @throw [NSException exceptionWithName:@"Can't save to disk"
                                       reason:@"The path must be convertible to UTF8"
                                     userInfo:nil];
    _native->save(path_c);
}

- (void)load:(NSString*)path {
    char const *path_c = [path UTF8String];
    if (!path_c)
        @throw [NSException exceptionWithName:@"Can't load from disk"
                                       reason:@"The path must be convertible to UTF8"
                                     userInfo:nil];
    _native->load(path_c);
}

- (void)view:(NSString*)path {
    char const *path_c = [path UTF8String];
    if (!path_c)
        @throw [NSException exceptionWithName:@"Can't view from disk"
                                       reason:@"The path must be convertible to UTF8"
                                     userInfo:nil];
    _native->view(path_c);
}

@end
