#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, USearchScalar) {
   USearchScalarF32,
   USearchScalarF16,
   USearchScalarF64,
   USearchScalarF8,
   USearchScalarB1
};

typedef NS_ENUM(NSUInteger, USearchMetric) {
   USearchMetricUnknown,
   USearchMetricIP,
   USearchMetricCos,
   USearchMetricL2sq,
   USearchMetricPearson,
   USearchMetricHaversine,
   USearchMetricJaccard,
   USearchMetricHamming,
   USearchMetricTanimoto,
   USearchMetricSorensen
};

API_AVAILABLE(ios(13.0), macos(10.15), tvos(13.0), watchos(6.0))
@interface USearchIndex : NSObject

@property (readonly) UInt32 dimensions;
@property (readonly) UInt32 connectivity;
@property (readonly) UInt32 expansionAdd;
@property (readonly) UInt32 expansionSearch;

@property (readonly) UInt32 length;
@property (readonly) UInt32 capacity;
@property (readonly) Boolean isEmpty;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)make:(USearchMetric)metric dimensions:(UInt32)dimensions connectivity:(UInt32)connectivity quantization:(USearchScalar)quantization NS_SWIFT_NAME(make(metric:dimensions:connectivity:quantization:));

- (void)addSingle:(UInt32)label
           vector:(Float32 const *_Nonnull)vector NS_SWIFT_NAME(addSingle(label:vector:));

- (UInt32)searchSingle:(Float32 const *_Nonnull)vector
                 count:(UInt32)count
                labels:(UInt32 *_Nullable)labels
             distances:(Float32 *_Nullable)distances NS_SWIFT_NAME(searchSingle(vector:count:labels:distances:));

- (void)addDouble:(UInt32)label
            vector:(Float64 const *_Nonnull)vector NS_SWIFT_NAME(addDouble(label:vector:));

- (UInt32)searchDouble:(Float64 const *_Nonnull)vector
                  count:(UInt32)count
                 labels:(UInt32 *_Nullable)labels
              distances:(Float32 *_Nullable)distances NS_SWIFT_NAME(searchDouble(vector:count:labels:distances:));

- (void)addHalf:(UInt32)label
              vector:(void const *_Nonnull)vector NS_SWIFT_NAME(addHalf(label:vector:));

- (UInt32)searchHalf:(void const *_Nonnull)vector
                    count:(UInt32)count
                   labels:(UInt32 *_Nullable)labels
                distances:(Float32 *_Nullable)distances NS_SWIFT_NAME(searchHalf(vector:count:labels:distances:));

- (void)save:(NSString *)path NS_SWIFT_NAME(save(path:));
- (void)load:(NSString *)path NS_SWIFT_NAME(load(path:));
- (void)view:(NSString *)path NS_SWIFT_NAME(view(path:));
- (void)clear NS_SWIFT_NAME(clear());

@end

NS_ASSUME_NONNULL_END
