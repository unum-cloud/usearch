#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, Quantization) {
    kSinglePrecision,
    kHalfPrecision,
    kDoublePrecision
};

@interface Index: NSObject

@property (readonly) UInt dimensions;
@property (readonly) UInt connectivity;
@property (readonly) UInt expansion_add;
@property (readonly) UInt expansion_search;

@property (readonly) UInt length;
@property (readonly) UInt capacity;
@property (readonly) Boolean isEmpty;

- (instancetype)init
NS_UNAVAILABLE;

+ (instancetype)indexIP:(UInt)dimensions connectivity:(UInt)connectivity NS_SWIFT_NAME(indexIP(dimensions:connectivity:));
+ (instancetype)indexL2:(UInt)dimensions connectivity:(UInt)connectivity NS_SWIFT_NAME(l2(dimensions:connectivity:));
+ (instancetype)indexHaversine:(UInt)connectivity NS_SWIFT_NAME(indexHaversine(connectivity:));

- (void)addSingle:(UInt32)label
           vector:(Float32 const* _Nonnull)vector
NS_SWIFT_NAME(addSingle(label:vector:));

- (UInt)searchSingle:(Float32 const* _Nonnull)vector
               count:(UInt)count
              labels:(UInt32* _Nullable)labels
           distances:(Float32* _Nullable)distances
NS_SWIFT_NAME(searchSingle(vector:count:labels:distances:));

- (void)addPrecise:(UInt32)label
            vector:(Float64 const* _Nonnull)vector
NS_SWIFT_NAME(addPrecise(label:vector:));

- (UInt)searchPrecise:(Float64 const* _Nonnull)vector
                count:(UInt)count
               labels:(UInt32* _Nullable)labels
            distances:(Float32* _Nullable)distances
NS_SWIFT_NAME(searchPrecise(vector:count:labels:distances:));

- (void)addImprecise:(UInt32)label
              vector:(void const* _Nonnull)vector
NS_SWIFT_NAME(addImprecise(label:vector:));

- (UInt)searchImprecise:(void const* _Nonnull)vector
                  count:(UInt)count
                 labels:(UInt32* _Nullable)labels
              distances:(Float32* _Nullable)distances
NS_SWIFT_NAME(searchImprecise(vector:count:labels:distances:));


- (void)save:(NSString*)path NS_SWIFT_NAME(dump(path:));
- (void)load:(NSString*)path NS_SWIFT_NAME(load(path:));
- (void)view:(NSString*)path NS_SWIFT_NAME(view(path:));
- (void)clear NS_SWIFT_NAME(clear());

@end

NS_ASSUME_NONNULL_END
