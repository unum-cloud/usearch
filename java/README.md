# USearch for Java

## Installation

```xml
<dependency>
  <groupId>cloud.unum</groupId>
  <artifactId>usearch</artifactId>
  <version>2.19.17</version>
</dependency>
```

Add that snippet to your `pom.xml` and hit `mvn install`.

## Quickstart

```java
import cloud.unum.usearch.Index;

public class Main {
    public static void main(String[] args) {
        try (Index index = new Index.Config()
                .metric(Index.Metric.COSINE)              // Or "cos"
                .quantization(Index.Quantization.FLOAT32) // Or "f32"
                .dimensions(3)
                .capacity(100)
                .build()) {
            
            // Add to Index
            float[] vector = {0.1f, 0.2f, 0.3f};
            index.add(42L, vector);

            // Search
            long[] keys = index.search(new float[]{0.1f, 0.2f, 0.3f}, 10);
            for (long key : keys) {
                System.out.println("Found key: " + key);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## Serialization

To save and load the index from disk, use the following methods:

```java
index.save("index.usearch");
index.load("index.usearch");
index.view("index.usearch");
```

## Extracting, Updating, and Removing Values

It is generally not recommended to use HNSW indexes in case of frequent removals or major distribution shifts.
For small updates, you can use the following methods:

```java
float[] vector = index.get(42L);
boolean removed = index.remove(42L);
boolean renamed = index.rename(43L, 42L);
```

To obtain metadata:

```java
long size = index.size();
long capacity = index.capacity();
long dimensions = index.dimensions();
long connectivity = index.connectivity();
```
