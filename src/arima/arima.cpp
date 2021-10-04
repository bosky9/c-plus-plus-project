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