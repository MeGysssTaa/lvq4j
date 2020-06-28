# About
**LVQ4J** is a basic implementation of the [LVQ](https://en.wikipedia.org/wiki/Learning_vector_quantization) (Learning Vector Quantization) [prototype-based](https://en.wikipedia.org/wiki/Prototype) [supervised](https://en.wikipedia.org/wiki/Supervised_learning) [classification](https://en.wikipedia.org/wiki/Statistical_classification) [algorithm](https://en.wikipedia.org/wiki/Algorithm) written in **Java**, and an accompanying library for its easier use and setup.


# Heads up
**[I am](https://github.com/MeGysssTaa)** **not** a professional data scientist in any way. I created this library solely for my own small research purposes in the machine learning field according to some publicly available papers, articles, and tutorials.


> For this reason, I hereby state that **I cannot guarantee that this implementation is 100% accurate and will always work as expected**. Use LVQ4J in your projects **on your own risk**. 


# Contributing
* If you believe something is wrong with my LVQ implementation, or if you are having troubles using the API, **[please open an issue](https://github.com/MeGysssTaa/lvq4j/issues)**.
* If you want to make direct changes to the code of the library or the LVQ implementation itself, **[don't hesitate to make a pull request](https://github.com/MeGysssTaa/lvq4j/pulls)**!


# Why LVQ4J?
The main intention of LVQ4J is to provide a simple, user-friendly, and, most importantly, **lightweight** API for creating, training and using Learning Vector Quantization algorithms for classification (prediction) purposes. It might not be as optimized, as fast, or as powerful as other libraries, but it is a considerably good starting point for data science *beginners*. The code is pretty **small**, **easy to understand**, and is **well-documented**.
> If you are looking for a robust and/or GPU-optimized machine learning library, then you are wrong here. *Otherwise, if you're just a data newbie who would like to get started with LVQ, then you will probably love this library.*


# Features
* Basic implementation of the **[LVQ model](https://en.wikipedia.org/wiki/Learning_vector_quantization)** in pure Java;
* a variety of built-in **[input normalization](https://en.wikipedia.org/wiki/Normalization_(statistics))** functions;
* several premade **weights initialization** strategies;
* many default **[distance metrics](https://en.wikipedia.org/wiki/Metric_(mathematics))**;
* comparably high **level of abstraction** for beginners, yet with deep access to the neural network **at its lowest level** for experienced users;
* LVQ4J is **extremely lightweight** — the library itself is small, and the only dependency is **`Slf4j (log4j2)`**, which is **not _required_** thanks to a default *fallback* logger implementation.


# LVQ vs k-nn vs Deep Learning
In a nutshell, **LVQ** is an **"eagerly-learning"** variant of **[k-nn](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)**. LVQ is a **neural network**, whereas k-nn **is not**. It takes pretty long for an LVQ model to train, however, the performance of its predictions is a lot better compared to k-nn that has to do its CPU-heavy tricks on **every** classification due to its **"lazy"** learning nature. Moreover, LVQ can work with accuracy similar to that of a k-nn even with a significantly smaller amount of train data.


Nevertheless, LVQ is still one of the simplest neural network algorithms. In most cases its sole advantage over **[deep learning](https://en.wikipedia.org/wiki/Deep_learning)** (e.g. **[RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)** or **[SVM](https://en.wikipedia.org/wiki/Support_vector_machine)**) algorithms is that it is very **easy to implement** and setup for instant use. Compared to other **neural networks**, one does not have to have a lot of specific knowledge and experience in order to work with an LVQ model.


# Usage
## Maven
```xml
<repositories>
    <repository>
        <id>reflex.public</id>
        <name>Public Reflex Repository</name>
        <url>https://archiva.reflex.rip/repository/public/</url>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>me.darksidecode.lvq4j</groupId>
        <artifactId>lvq4j</artifactId>
        <version>1.0.3</version>
    </dependency>
</dependencies>
```


## Gradle
```groovy
repositories {
    maven {
        name 'Public Reflex Repository'
        url 'https://archiva.reflex.rip/repository/public/'
    }
}

dependencies {
    implementation group: 'me.darksidecode.lvq4j', name: 'lvq4j', version: '1.0.3'
}
```


# Examples
* [Iris Data Set](https://github.com/MeGysssTaa/lvq4j-example-iris)

> Using **LVQ4J** in an own project? Want it to be listed here? **[Feel free to make a pull request!](https://github.com/MeGysssTaa/lvq4j/pulls)**


# Bulding
```bash
git clone https://github.com/MeGysssTaa/lvq4j
cd lvq4j
./gradlew build
```


# License
**[Apache License 2.0](https://github.com/MeGysssTaa/lvq4j/blob/master/LICENSE)**
