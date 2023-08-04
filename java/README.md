# USearch for Java

## Installation

```xml
<dependency>
  <groupId>cloud.unum</groupId>
  <artifactId>usearch</artifactId>
  <version>0.2.3</version>
</dependency>
```

Add that snippet to your `pom.xml` and hit `mvn install`.

## Quickstart

```java
Index index = new Index.Config().metric("cos").dimensions(2).build();
float vec[] = {10, 20};
index.add(42, vec);
int[] keys = index.search(vec, 5);
```
