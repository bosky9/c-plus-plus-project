#include "arima/arima.hpp"

#include "data_check.hpp"
#include "families/family.hpp"

ARIMA::ARIMA(const std::vector<double>& data, const std::vector<double>& index, size_t ar, size_t ma, size_t integ, const Family& family) :
      TSM{"ARIMA"},
      _ar{ar},
      _ma{ma},
      _integ{integ}
{
    // Latent Variable information
    _z_no = _ar + _ma + 2;
    _max_lag = std::max(_ar, _ma);
    _z_hide = 0;
    _supported_methods = {"MLE", "PML", "Laplace", "M-H", "BBVI"};
    _default_method = "MLE";
    _multivariate_model = false;

    // Format the data
    CheckedData c_data = data_check(data, index);
    _data = c_data.transformed_data;
    _data_name = c_data.data_name;
    _index = c_data.data_index;

    // Difference data
    for (size_t order{0}; order < _integ; order++)
        _data = diff(_data);
    _data_name.at(0) = "Differenced " + _data_name.at(0);

    _x = ar_matrix();
    create_latent_variables();

    _family = family;
    FamilyAttributes fa = family.setup(); // TODO: Creare il metodo virtual in family che ritorna delle info nulle
    _model_name_short = fa.name;
    _link = fa.link;
    _scale = fa.scale;
    _shape = fa.shape;
    _skewness = fa.skewness;
    _mean_transform = fa.mean_transform;
    _cythonized = fa.cythonized;
    _model_name = _model_name_short + " ARIMA(" + std::to_string(_ar) + "," + std::to_string(_integ) + "," + std::to_string(_ma) + ")";

    // Build any remaining latent variables that are specific to the family chosen
    // TODO: continue
}

ARIMA::ARIMA(const std::map<std::string, std::vector<double>>& data, const std::vector<double>& index, const std::string& target, size_t ar, size_t ma, size_t integ, const Family& family) :
      TSM{"ARIMA"},
      _ar{ar},
      _ma{ma},
      _integ{integ}
{
    // Latent Variable information
    _z_no = _ar + _ma + 2;
    _max_lag = std::max(_ar, _ma);
    _z_hide = 0;
    _supported_methods = {"MLE", "PML", "Laplace", "M-H", "BBVI"};
    _default_method = "MLE";
    _multivariate_model = false;

    // Format the data
    CheckedData c_data = data_check(data, index, target);
    _data = c_data.transformed_data;
    _data_name = c_data.data_name;
    _index = c_data.data_index;

    // TODO: continue
}

std::tuple<double, double, double> ARIMA::get_scale_and_shape(Eigen::VectorXd transformed_lvs) {
    double model_shape      = 0;
    double model_scale      = 0;
    double model_skewness   = 0;

    if (_scale){
        if(_shape){
            model_shape = transformed_lvs(Eigen::last);
            model_scale = transformed_lvs(Eigen::last-1);
        }
        else
            model_scale = transformed_lvs(Eigen::last);
    }

    if (_skewness)
        model_skewness = transformed_lvs(Eigen::last-2);

    return std::make_tuple(model_scale, model_shape, model_skewness);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> ARIMA::get_scale_and_shape_sim(Eigen::MatrixXd transformed_lvs) {
    Eigen::VectorXd model_shape(transformed_lvs.cols())     = Eigen::VectorXd::Zero(transformed_lvs.cols());
    Eigen::VectorXd model_scale(transformed_lvs.cols())     = Eigen::VectorXd::Zero(transformed_lvs.cols());
    Eigen::VectorXd model_skewness(transformed_lvs.cols())  = Eigen::VectorXd::Zero(transformed_lvs.cols());

    if (_scale){
        if (_shape){
            // Apply trasform() to every element inside the matrix last row
            model_shape(Eigen::first, Eigen::last)= transformed_lvs(Eigen::last, Eigen::all);
            std::transform(model_shape.begin(), model_shape.end(), model_shape,
                    _latent_variables.get_z_list().back().get_prior()->get_transform());

            model_scale(Eigen::first, Eigen::last)= transformed_lvs(Eigen::last-1, Eigen::all);
            std::transform(model_scale.begin(), model_scale.end(), model_scale,
                           _latent_variables.get_z_list().back().get_prior()->get_transform());
        }
        else{
            model_scale(Eigen::first, Eigen::last)= transformed_lvs(Eigen::last, Eigen::all);
            std::transform(model_scale.begin(), model_scale.end(), model_scale,
                           _latent_variables.get_z_list().back().get_prior()->get_transform());
        }
    }

    if (_skewness){
        model_skewness(Eigen::first, Eigen::last) = transformed_lvs(Eigen::last-2, Eigen::all);
        std::transform(model_skewness.begin(), model_skewness.end(), model_skewness,
                       _latent_variables.get_z_list().back().get_prior()->get_transform());
    }

    return std::move(std::make_tuple(model_scale, model_shape, model_skewness));

}