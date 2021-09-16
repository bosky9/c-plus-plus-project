#include "../../include/families/flat.hpp"

Flat::Flat(const std::string& transform) : Family{transform}, _covariance_prior{false} {}

Flat::Flat(const Flat& flat) : Family(flat) {
    _covariance_prior = flat._covariance_prior;
}

Flat::Flat(Flat&& flat) noexcept : Family(std::move(flat)) {
    _covariance_prior      = flat._covariance_prior;
    flat._covariance_prior = false;
}

Flat& Flat::operator=(const Flat& flat) {
    if (this == &flat)
        return *this;
    Family::operator  =(flat);
    _covariance_prior = flat._covariance_prior;
    return *this;
}

Flat& Flat::operator=(Flat&& flat) noexcept {
    _covariance_prior      = flat._covariance_prior;
    flat._covariance_prior = false;
    Family::operator       =(std::move(flat));
    return *this;
}

bool operator==(const Flat& flat1, const Flat& flat2) {
    return is_equal(flat1, flat2) && flat1._covariance_prior == flat2._covariance_prior;
}

double Flat::logpdf(double mu) {
    return 0.0;
}

bool Flat::get_covariance_prior() const {
    return _covariance_prior;
}

std::string Flat::get_name() const {
    return "Flat";
}

std::string Flat::get_z_name() const {
    return "n/a (non-informative)";
}

Family* Flat::clone() const {
    return new Flat(*this);
}