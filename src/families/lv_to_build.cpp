#include "families/lvs_to_build.hpp"

Lv_to_build::Lv_to_build(const std::string& name, Flat* flat, Normal* normal, double value) : _name{name} {
    _flat.reset(flat);
    _normal.reset(normal);
    _value = value;
}

Lv_to_build::Lv_to_build(const Lv_to_build& build) {
    _name = build._name;
    _flat.reset(build._flat.get());
    _normal.reset(build._normal.get());
    _value = build._value;
}

Lv_to_build::Lv_to_build(Lv_to_build&& build) = default;

Lv_to_build& Lv_to_build::operator=(const Lv_to_build& build) {
    if (this == &build)
        return *this;
    _name = build._name;
    _flat.reset(build._flat.get());
    _normal.reset(build._normal.get());
    _value = build._value;
    return *this;
}

Lv_to_build& Lv_to_build::operator=(Lv_to_build&& build) = default;

Lv_to_build::~Lv_to_build() = default;