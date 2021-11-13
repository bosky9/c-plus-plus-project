/**
 * @file flat.cpp
 * @author Bodini Alessia, Boschi Federico e Cinquetti Ettore
 * @date November, 2021
 */

#include "families/flat.hpp"

Flat::Flat(const std::string& transform) : Family{transform}, _covariance_prior{false} {}

Flat::Flat(const Flat& flat) = default;

Flat::Flat(Flat&& flat) noexcept = default;

Flat& Flat::operator=(const Flat& flat) = default;

Flat& Flat::operator=(Flat&& flat) noexcept = default;

bool operator==(const Flat& flat1, const Flat& flat2) {
    return flat1._transform_name == flat2._transform_name && flat1._itransform_name == flat2._itransform_name &&
           flat1._covariance_prior == flat2._covariance_prior;
}

double Flat::logpdf([[maybe_unused]] double mu) {
    return 0.0;
}

// Get methods -----------------------------------------------------------------------------------------------------

std::string Flat::get_name() const {
    return "Flat";
}

std::string Flat::get_z_name() const {
    return "n/a (non-informative)";
}

// Clone function --------------------------------------------------------------------------------------------------

Family* Flat::clone() const {
    return new Flat(*this);
}