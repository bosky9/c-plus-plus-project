# PyFlux project - C++ version

As defined in the original project:
> PyFlux is an open source time series library for Python. The library has a good array of modern time series models, as well as a flexible array of inference options (frequentist and Bayesian) that can be applied to these models. By combining breadth of models with breadth of inference, PyFlux allows for a probabilistic approach to time series modelling.

Here's the [link](https://github.com/RJT1990/pyflux) to the original project.

## Table of Contents

1. [About the Project](#about-the-project)
1. [Project Status](#project-status)
1. [Getting Started](#getting-started)
    1. [Dependencies](#dependencies)
    1. [Building](#building)
    2. [Running Tests](#running-tests)
        1. [Other Tests](#other-tests)
    1. [Installation](#installation)
    1. [Usage](#usage)
1. [Release Process](#release-process)
    1. [Versioning](#versioning)
    1. [Payload](#payload)
1. [How to Get Help](#how-to-get-help)
1. [Contributing](#contributing)
1. [Further Reading](#further-reading)
1. [License](#license)
1. [Authors](#authors)
1. [Acknowledgments](#acknowledgements)

## About the Project

From the original project we took the section concerning the (basic) [ARIMA model](https://pyflux.readthedocs.io/en/latest/arima.html) and its usage with a Normal family of functions (see [here](https://pyflux.readthedocs.io/en/latest/families.html) all the available families) and we translated all the files in C++. 

Here's an example of what you can do with the ARIMA class we defined: 
```c++
// Import data from some sunspot values 
utils::DataFrame data = utils::parse_csv("../data/sunspot.year.csv");
// Define a new ARIMA object
ARIMA model{data, 1, 1, 0, Normal(0, 3)};
// Fit the model using BBVI method 
std::optional<Eigen::MatrixXd> opt_matrix{std::nullopt};
Results* x{model.fit("BBVI", opt_matrix, 100, 10000, "RMSProp", 12, std::nullopt, true, 1e-03, std::nullopt,
                             true)};
// Predict values from the model 
utils::DataFrame predictions = model.predict(10, true);
```

**[Back to top](#table-of-contents)**

## Project Status

Currently our project doesn't cover all the original work. As said before, we only defined all the necessary files to work with an ARIMA model using exclusively the Normal family, but so many other families and models are still waiting to be translated. 

Besides that, in the current status our program works perfectly using both GCC and Clang compilers. 

**[Back to top](#table-of-contents)**

## Getting Started

To compile and try the library yourself you just need to use the `make` command from the project folder. All instructions for building and testing are available in the CMakeLists.txt file. 

### Dependencies

This project highly relies on some external libraries, in particular on the [Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page), used for defining most of the vectors and matrices originally created using the [NumPy library](https://numpy.org/) in Python.
Other than that, we also used [LBFGSpp library](https://github.com/yixuan/LBFGSpp) to define optimization functions and [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) to recreate the same graphs available in the Python project. For this last mention an installation of Python3 is is necessary, since `matplotlibcpp.hpp` call directly its functions. 

All the libraries mentioned above are already inside the project in the *third_party* directory, without any need to be downloaded. 

Finally we also used [Catch2](https://github.com/catchorg/Catch2) to write test for every class, that can be seen in the *test* folder. 

```
Examples should be included
```

### Getting the Source

This project is [hosted on GitHub](https://github.com/bosky9/c-plus-plus-project). You can clone this project directly using this command:

```
git clone https://github.com/bosky9/c-plus-plus-project.git
```

### Building

To build the project you simply need to call `cmake` on the project folder and after that the `make` command. A new executable *tests* will then be found inside the *bin* directory. 

```
cmake . -B build
make -C build
```

### Running Tests

To run all the test you just need to call the proper command:

```
make test -C build
```

#### Other Tests

For formatting [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) with the *.clang_format* file is used, while as static analyzer we used [cppcheck](https://cppcheck.sourceforge.io/). 

### Installation

Instructions for how to install your project's build artifacts

```
Examples should be included
```

### Usage

Instructions for using your project. Ways to run the program, how to include it in another project, etc.

```
Examples should be included
```

If your project provides an API, either provide details for usage in this document or link to the appropriate API reference documents

**[Back to top](#table-of-contents)**

## Release Process

Talk about the release process. How are releases made? What cadence? How to get new releases?

### Versioning

This project uses [Semantic Versioning](http://semver.org/). For a list of available versions, see the [repository tag list](https://github.com/your/project/tags).

### Payload

**[Back to top](#table-of-contents)**

## How to Get Help

Provide any instructions or contact information for users who need to get further help with your project.

## Contributing

Provide details about how people can contribute to your project. If you have a contributing guide, mention it here. e.g.:

We encourage public contributions! Please review [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and development process.

**[Back to top](#table-of-contents)**

## Further Reading

Provide links to other relevant documentation here

**[Back to top](#table-of-contents)**

## License

Copyright (c) 2021 Embedded Artistry LLC

This project is licensed under the XXXXXX License - see [LICENSE.md](LICENSE.md) file for details.

**[Back to top](#table-of-contents)**

## Authors

* **[Alessia Bodini](https://github.com/alessiabodini)** - VR451051
* **[Federisco Boschi](https://github.com/bosky9)** - VR
* **[Ettore Cinquetti](https://github.com/e5ti)** - VR

**[Back to top](#table-of-contents)**

## Acknowledgments

Provide proper credits, shout-outs, and honorable mentions here. Also provide links to relevant repositories, blog posts, or contributors worth mentioning.

**[Back to top](#table-of-contents)**