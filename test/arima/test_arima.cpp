#include <catch2/catch_test_macros.hpp>
#include <lbfgspp/LBFGS.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>

#include <optional>
#include "arima/arima.hpp"

namespace py = pybind11;

py::object prova(py::function minimize, py::function fun, Eigen::VectorXd& beta) {
    return minimize(fun, beta);
}

TEST_CASE("Test normal_neg_loglik", "[ARIMA]") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    //ARIMA my_arima{data, 2, 2};
    //ARIMA other_arima{data, 3 ,3};

    py::scoped_interpreter guard{};

    //Eigen::VectorXd beta                       = my_arima.get_phi();
    Eigen::VectorXd beta(6); beta << 23,2,3,5,11,24;
    Eigen::VectorXd beta2(6); beta2 << 23,2,3,5,11,24;
    std::function<double(Eigen::VectorXd)> fun = [](const Eigen::VectorXd& x) { return 2*x[0]*x[0]; };

    py::function minimize = py::module::import("scipy.optimize").attr("minimize");
    py::function py_fun   = py::cast(fun);
    py::object p          = prova(minimize, py_fun, beta);
    py::object p2          = prova(minimize, py_fun, beta2);

    Eigen::VectorXd x     = p.attr("x").cast<Eigen::VectorXd>();
    bool success          = p.attr("success").cast<bool>();
    int niter             = p.attr("nit").cast<int>();
    double f              = p.attr("fun").cast<double>();

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "Success: " << success << std::endl;
    std::cout << "Beta: " << beta << '\n';
    std::cout << "f(x) = " << f << std::endl;

    /*
    TEST_CASE("Tests with no AR or MA terms", "[fit]") {
        ARIMA model{data, 0, 0};
        Results* x{model.fit()};
        REQUIRE(model.get_latent_variables().get_z_list().size() == 2);
        std::vector<LatentVariable> lvs{model.get_latent_variables().get_z_list()};

        size_t nan{0};
        for (size_t i{0}; i < lvs.size(); i++) {
            if (!lvs[i].get_value().has_value())
                nan++;
        }
        REQUIRE(nan == 0);

        delete x;
    }

*/
        ARIMA model{data, 2, 2, minimize};
        Results* res{model.fit()};
        REQUIRE(model.get_latent_variables().get_z_list().size() == 6);
        std::vector<LatentVariable> lvs{model.get_latent_variables().get_z_list()};

        size_t nan{0};
        for (size_t i{0}; i < lvs.size(); i++) {
            if (!lvs[i].get_value().has_value())
                nan++;
        }
        REQUIRE(nan == 0);
        REQUIRE(model.predict(5).data.at(0).size() == 5);
        REQUIRE(model.predict_is(5).data.at(0).size() == 5);

        DataFrame test_df = model.predict(5);

        for (auto& vec : test_df.data)
            for (auto& elem : vec)
                REQUIRE(!std::isnan(elem));

        DataFrame predictions = model.predict(10, false);
        DataFrame predictions_is = model.predict_is(10, false);

        // forecasted_values, prediction_01, prediction_05, prediction_95, prediction_99
        /*
        for (int i = 4; i > 1; i--) {
            double sup_min_element = *min_element(std::begin(predictions.data.at(i)), std::end(predictions.data.at(i)));
            double inf_max_element =
                    *min_element(std::begin(predictions.data.at(i - 1)), std::end(predictions.data.at(i - 1)));
            REQUIRE(sup_min_element >= inf_max_element);
        }

        double sup_min_element = *min_element(std::begin(predictions.data.at(3)), std::end(predictions.data.at(3)));
        double inf_max_element = *min_element(std::begin(predictions.data.at(0)), std::end(predictions.data.at(0)));
        REQUIRE(sup_min_element >= inf_max_element);
         */

        /*
        REQUIRE(!(std::adjacent_find(predictions.data.begin(), predictions.data.end(), std::not_equal_to<>()) ==
        predictions.data.end()));

        REQUIRE(!(std::adjacent_find(predictions_is.data.begin(), predictions_is.data.end(), std::not_equal_to<>()) ==
        predictions_is.data.end()));
         */

        delete res;
}
/*
    // @TODO: ARIMA with 2 AR 2 MA (check if previous two work)

TEST_CASE("Test prediction length", "[fit]") {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, 1);
        std::vector<double> data(100, 0);
        for (size_t i{1}; i < 100; i++)
            data[i] = 0.9 * data[i - 1] + distribution(generator);

        ARIMA model{data, 2, 2};
        Results* x{model.fit()};

        REQUIRE(model.predict(5).data.at(0).size() == 5);

        delete x;
    }

TEST_CASE("Test prediction IS length", "[fit]") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    ARIMA model{data, 2, 2};
    Results* x{model.fit()};

    REQUIRE(model.predict_is(5).data.at(0).size() == 5);
    py::finalize_interpreter();

    delete x;
}

TEST_CASE("Test that the predictions are not nans") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    ARIMA model{data, 2, 2};
    Results* x{model.fit()};
    py::finalize_interpreter();

    DataFrame test_df = model.predict(5);

    for (auto& vec : test_df.data)
        for (auto& elem : vec)
            REQUIRE(!std::isnan(elem));
    delete x;
}

TEST_CASE("Test that the predictions IS are not nans") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    ARIMA model{data, 2, 2};
    Results* x{model.fit()};
    py::finalize_interpreter();

    DataFrame test_df = model.predict_is(5);

    for (auto& vec : test_df.data)
        for (auto& elem : vec)
            REQUIRE(!std::isnan(elem));
    delete x;
}

TEST_CASE("Test predictions not having constant values") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    ARIMA model{data, 2, 2};
    Results* x{model.fit()};
    py::finalize_interpreter();

    DataFrame predictions = model.predict(10, false);

    REQUIRE(!(std::adjacent_find(predictions.data.begin(), predictions.data.end(), std::not_equal_to<>()) ==
              predictions.data.end()));
    delete x;
}

TEST_CASE("Test IS predictions not having constant values") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);

    ARIMA model{data, 2, 2};
    Results* x{model.fit()};
    py::finalize_interpreter();

    DataFrame predictions = model.predict_is(10, false);
    REQUIRE(!(std::adjacent_find(predictions.data.begin(), predictions.data.end(), std::not_equal_to<>()) ==
              predictions.data.end()));
        delete x;
}

TEST_CASE("Tests prediction intervals are ordered correctly") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);
        ARIMA model{data, 2, 2};
        Results* x{model.fit()};
        DataFrame predictions = model.predict(10, false);

        // forecasted_values, prediction_01, prediction_05, prediction_95, prediction_99
        for (int i = 4; i > 1; i--) {
            double sup_min_element = *min_element(std::begin(predictions.data.at(i)), std::end(predictions.data.at(i)));
            double inf_max_element =
                    *min_element(std::begin(predictions.data.at(i - 1)), std::end(predictions.data.at(i - 1)));
            REQUIRE(sup_min_element >= inf_max_element);
        }

        double sup_min_element = *min_element(std::begin(predictions.data.at(3)), std::end(predictions.data.at(3)));
        double inf_max_element = *min_element(std::begin(predictions.data.at(0)), std::end(predictions.data.at(0)));
        REQUIRE(sup_min_element >= inf_max_element);

        delete x;
    }

    //@TODO: se giusto, fai tutti gli altri (fino a riga 161)

TEST_CASE("Test sampling function") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);
    ARIMA model{data, 2, 2};
    auto op = std::nullopt;
    // ok cosa
    std::optional<Eigen::MatrixXd>& op_matrix = (std::optional<Eigen::MatrixXd>&) std::nullopt;
    Results* x{model.fit("BBVI", false, op_matrix, 100, op, op, op, op, op, op, op, true)};
    Eigen::MatrixXd sample = model.sample(100);

    REQUIRE(sample.rows() == 100);
    REQUIRE(sample.cols() == data.size() - 2);

    delete x;
    }

TEST_CASE("Test ppc value") {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1);
    std::vector<double> data(100, 0);
    for (size_t i{1}; i < 100; i++)
        data[i] = 0.9 * data[i - 1] + distribution(generator);
        ARIMA model{data, 2, 2};
        auto op                                   = std::nullopt;
        std::optional<Eigen::MatrixXd>& op_matrix = (std::optional<Eigen::MatrixXd>&) std::nullopt;
        Results* x{model.fit("BBVI", false, op_matrix, 100, op, op, op, op, op, op, op, true)};
        double p_value = model.ppc();

        REQUIRE(p_value >= 0.0);
        REQUIRE(p_value <= 1.0);

        delete x;
}
*/