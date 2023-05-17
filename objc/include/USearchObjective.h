#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, Quantization) {
    kSinglePrecision,
    kHalfPrecision,
    kDoublePrecision
};

API_AVAILABLE(ios(13.0), macos(10.15), tvos(13.0), watchos(6.0))
@interface Index: NSObject

@property (readonly) UInt32 dimensions;
@property (readonly) UInt32 connectivity;
@property (readonly) UInt32 expansion_add;
@property (readonly) UInt32 expansion_search;

@property (readonly) UInt32 length;
@property (readonly) UInt32 capacity;
@property (readonly) Boolean isEmpty;

- (instancetype)init
NS_UNAVAILABLE;

+ (instancetype)indexIP:(UInt32)dimensions connectivity:(UInt32)connectivity NS_SWIFT_NAME(indexIP(dimensions:connectivity:));
+ (instancetype)indexL2:(UInt32)dimensions connectivity:(UInt32)connectivity NS_SWIFT_NAME(l2(dimensions:connectivity:));
+ (instancetype)indexHaversine:(UInt32)connectivity NS_SWIFT_NAME(indexHaversine(connectivity:));

- (void)addSingle:(UInt32)label
           vector:(Float32 const* _Nonnull)vector
NS_SWIFT_NAME(addSingle(label:vector:));

- (UInt32)searchSingle:(Float32 const* _Nonnull)vector
                 count:(UInt32)count
                labels:(UInt32* _Nullable)labels
             distances:(Float32* _Nullable)distances
NS_SWIFT_NAME(searchSingle(vector:count:labels:distances:));

- (void)addPrecise:(UInt32)label
            vector:(Float64 const* _Nonnull)vector
NS_SWIFT_NAME(addPrecise(label:vector:));

- (UInt32)searchPrecise:(Float64 const* _Nonnull)vector
                  count:(UInt32)count
                 labels:(UInt32* _Nullable)labels
              distances:(Float32* _Nullable)distances
NS_SWIFT_NAME(searchPrecise(vector:count:labels:distances:));

- (void)addImprecise:(UInt32)label
              vector:(void const* _Nonnull)vector
NS_SWIFT_NAME(addImprecise(label:vector:));

- (UInt32)searchImprecise:(void const* _Nonnull)vector
                    count:(UInt32)count
                   labels:(UInt32* _Nullable)labels
                distances:(Float32* _Nullable)distances
NS_SWIFT_NAME(searchImprecise(vector:count:labels:distances:));


- (void)save:(NSString*)path NS_SWIFT_NAME(dump(path:));
- (void)load:(NSString*)path NS_SWIFT_NAME(load(path:));
- (void)view:(NSString*)path NS_SWIFT_NAME(view(path:));
- (void)clear NS_SWIFT_NAME(clear());

@end

NS_ASSUME_NONNULL_END
