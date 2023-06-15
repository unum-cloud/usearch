#import "USearchObjective.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#import <usearch/index_punned_dense.hpp>
#pragma clang diagnostic pop

using namespace unum::usearch;
using namespace unum;

using distance_t = punned_distance_t;
using punned_t = index_punned_dense_gt<UInt32>;
using shared_index_t = std::shared_ptr<punned_t>;

metric_kind_t to_native_metric(USearchMetric m) {
    switch (m) {
        case USearchMetricIP:
            return metric_kind_t::ip_k;
        case USearchMetricCos:
            return metric_kind_t::cos_k;
        case USearchMetricL2sq:
            return metric_kind_t::l2sq_k;
        case USearchMetricHamming:
            return metric_kind_t::hamming_k;
        case USearchMetricHaversine:
            return metric_kind_t::haversine_k;
        case USearchMetricJaccard:
            return metric_kind_t::jaccard_k;
        case USearchMetricPearson:
            return metric_kind_t::pearson_k;
        case USearchMetricSorensen:
            return metric_kind_t::sorensen_k;
        case USearchMetricTanimoto:
            return metric_kind_t::tanimoto_k;
        default:
            return metric_kind_t::unknown_k;
    }
}

scalar_kind_t to_native_scalar(USearchScalar m) {
    switch (m) {
        case USearchScalarF8:
            return scalar_kind_t::f8_k;
        case USearchScalarF16:
            return scalar_kind_t::f16_k;
        case USearchScalarF32:
            return scalar_kind_t::f32_k;
        case USearchScalarF64:
            return scalar_kind_t::f64_k;
        default:
            return scalar_kind_t::unknown_k;
    }
}

@interface USearchIndex ()

@property (readonly) shared_index_t native;

- (instancetype)initWithIndex:(shared_index_t)native;

@end

@implementation USearchIndex

- (instancetype)initWithIndex:(shared_index_t)native {
    self = [super init];
    _native = native;
    return self;
}

- (Boolean)isEmpty {
    return _native->size() != 0;
}

- (UInt32)dimensions {
    return static_cast<UInt32>(_native->dimensions());
}

- (UInt32)connectivity {
    return static_cast<UInt32>(_native->connectivity());
}

- (UInt32)length {
    return static_cast<UInt32>(_native->size());
}

- (UInt32)capacity {
    return static_cast<UInt32>(_native->capacity());
}

- (UInt32)expansionAdd {
    return static_cast<UInt32>(_native->expansion_add());
}

- (UInt32)expansionSearch {
    return static_cast<UInt32>(_native->expansion_search());
}

+ (instancetype)make:(USearchMetric)metric dimensions:(UInt32)dimensions connectivity:(UInt32)connectivity quantization:(USearchScalar)quantization{
    std::size_t dims = static_cast<std::size_t>(dimensions);
    index_config_t config;

    config.connectivity = static_cast<std::size_t>(connectivity);
    shared_index_t ptr = std::make_shared<punned_t>(punned_t::make(dims, to_native_metric(metric), config, to_native_scalar(quantization)));
    return [[USearchIndex alloc] initWithIndex:ptr];
}

- (void)addSingle:(UInt32)label
           vector:(Float32 const *_Nonnull)vector {
    _native->add(label, vector);
}

- (UInt32)searchSingle:(Float32 const *_Nonnull)vector
                 count:(UInt32)wanted
                labels:(UInt32 *_Nullable)labels
             distances:(Float32 *_Nullable)distances {
    std::size_t found = _native->search(vector, static_cast<std::size_t>(wanted)).dump_to(labels, distances);

    return static_cast<UInt32>(found);
}

- (void)addDouble:(UInt32)label
            vector:(Float64 const *_Nonnull)vector {
    _native->add(label, (f64_t const *)vector);
}

- (UInt32)searchDouble:(Float64 const *_Nonnull)vector
                  count:(UInt32)wanted
                 labels:(UInt32 *_Nullable)labels
              distances:(Float32 *_Nullable)distances {
    std::size_t found = _native->search((f64_t const *)vector, static_cast<std::size_t>(wanted)).dump_to(labels, distances);

    return static_cast<UInt32>(found);
}

- (void)addHalf:(UInt32)label
              vector:(void const *_Nonnull)vector {
    _native->add(label, (f16_bits_t const *)vector);
}

- (UInt32)searchHalf:(void const *_Nonnull)vector
                    count:(UInt32)wanted
                   labels:(UInt32 *_Nullable)labels
                distances:(Float32 *_Nullable)distances {
    std::size_t found = _native->search((f16_bits_t const *)vector, static_cast<std::size_t>(wanted)).dump_to(labels, distances);

    return static_cast<UInt32>(found);
}

- (void)clear {
    _native->clear();
}

- (void)save:(NSString *)path {
    char const *path_c = [path UTF8String];

    if (!path_c) {
        @throw [NSException exceptionWithName:@"Can't save to disk"
                                       reason:@"The path must be convertible to UTF8"
                                     userInfo:nil];
    }

    _native->save(path_c);
}

- (void)load:(NSString *)path {
    char const *path_c = [path UTF8String];

    if (!path_c) {
        @throw [NSException exceptionWithName:@"Can't load from disk"
                                       reason:@"The path must be convertible to UTF8"
                                     userInfo:nil];
    }

    _native->load(path_c);
}

- (void)view:(NSString *)path {
    char const *path_c = [path UTF8String];

    if (!path_c) {
        @throw [NSException exceptionWithName:@"Can't view from disk"
                                       reason:@"The path must be convertible to UTF8"
                                     userInfo:nil];
    }

    _native->view(path_c);
}

@end
