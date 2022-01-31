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

From the original project we took the section concerning the (
basic) [ARIMA model](https://pyflux.readthedocs.io/en/latest/arima.html) and its usage with a Normal family of
functions (see [here](https://pyflux.readthedocs.io/en/latest/families.html) all the available families) and we
translated all the files in C++.

Here's an example of what you can do with the ARIMA class we defined:

```c++
// Import data from some sunspot values 
utils::DataFrame data = utils::parse_csv("../data/sunspot.year.csv");
// Define a new ARIMA object
ARIMA model{data, 1, 1, 0, Normal(0, 3)};
// Fit the model using BBVI method 
Results* x{model.fit("BBVI")};
// Predict values from the model 
utils::DataFrame predictions = model.predict(10, true);
```

**[Back to top](#table-of-contents)**

## Project Status

Currently our project doesn't cover all the original work. As said before, we only defined all the necessary files to
work with an ARIMA model using exclusively the Normal family, but so many other families and models are still waiting to
be translated.

Besides that, in the current status our program works perfectly using both GCC and Clang compilers.

**[Back to top](#table-of-contents)**

## Getting Started

To compile and try the library yourself you just need to use the `make` command from the project folder. All
instructions for building and testing are available in the CMakeLists.txt file.

### Dependencies

This project highly relies on some external libraries, in particular on
the [Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page), used for defining most of the vectors and
matrices originally created using the [NumPy library](https://numpy.org/) in Python. Other than that, we also
used [LBFGSpp library](https://github.com/yixuan/LBFGSpp) to define optimization functions
and [sciplot](https://sciplot.github.io/) to recreate the same graphs available in the Python project.

All the libraries mentioned above are already inside the project in the *third_party* directory, without any need to be
downloaded.

Finally we also used [Catch2](https://github.com/catchorg/Catch2) to write test for every class, that can be seen in
the *test* folder.

A test example done using Catch2 and the data from `sunspot.year.csv` is the following:

```c++
TEST_CASE("Test an ARIMA model with sunspot years data", "[ARIMA]") {
    utils::DataFrame data = utils::parse_csv("../data/sunspot.year.csv");

    // Tests on ARIMA model with 1 AR and 1 MA term that the latent variable list length is correct 
    // and that the estimated latent variables are not nan
    SECTION("Test an ARIMA model with 1 AR and 1 MA term", "[fit]") {
        ARIMA model{data, 1, 1, 0, Normal(0, 3)};
        Results* x{model.fit()};
        REQUIRE(model.get_latent_variables().get_z_list().size() == 4);

        std::vector<LatentVariable> lvs{model.get_latent_variables().get_z_list()};
        int64_t nan{std::count_if(lvs.begin(), lvs.end(),
                                  [](const LatentVariable& lv) { return !lv.get_value().has_value(); })};
        REQUIRE(nan == 0);

        delete x;
    }

```

### Getting the Source

This project is [hosted on GitHub](https://github.com/bosky9/c-plus-plus-project). You can clone this project directly
using this command:

```
git clone https://github.com/bosky9/c-plus-plus-project.git
```

### Building

To build the project you simply need to call `cmake` on the project folder and after that the `make` command. A new
executable *tests* will then be found inside the *bin* directory.

```
cmake -B build
make -C build
```

### Running Tests

To run all the test you just need to call the proper command:

```
make test -C build
```

To also activate _Valgrind_ you need to include the `memcheck` command in `ctest` from the build directory:

```
ctest -T memcheck -R testName
```

The `-R` let you specify a specific test to run with _Valgrind_ (we recommend using `TestSunposts`).

#### Other Tests

For formatting [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) with the *.clang_format* file is used, while
as static analyzers we used [cppcheck](https://cppcheck.sourceforge.io/)
and [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/).

### Installation

For moving the library in _bin_ folder:

```
make install -C build
```

**[Back to top](#table-of-contents)**

## Release Process

The project was started in August 2021 and finished in January 2022.

### Versioning

Currently only the first and current 1.0.0 version is available.

## Further Reading

[Here](data/report.pdf)'s a link the full report regarding this project.

## License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) file for details.

## Authors

* **[Alessia Bodini](https://github.com/alessiabodini)** - VR451051
* **[Federisco Boschi](https://github.com/bosky9)** - VR445479
* **[Ettore Cinquetti](https://github.com/e5ti)** - VR451823

## Acknowledgments

All the credits goes to the [PyFlux](https://github.com/RJT1990/pyflux) project.

**[Back to top](#table-of-contents)**
