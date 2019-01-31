// author: Justus Schock (justus.schock@rwth-aachen.de)

#include <torch/torch.h>
#include <vector>
#include <iostream>

at::Tensor shape_func_forward(at::Tensor shape_params,
                              at::Tensor shape_mean,
                              at::Tensor shape_components
){

    auto shapes = shape_mean.clone();
    at::IntList size_list = {shape_params.size(0)};

    auto size_list_shapes = size_list.vec();
    size_list_shapes.insert(std::end(size_list_shapes), std::begin(shapes.sizes().slice(1)), std::end(shapes.sizes()));
    shapes = shapes.expand(size_list_shapes);

    auto components = shape_components.clone();

    auto size_list_components = size_list.vec();
    size_list_components.insert(std::end(size_list_components), std::begin(components.sizes().slice(1)),
                                std::end(components.sizes()));

    auto weighted_components = components.mul(shape_params.expand_as(components));
;
    shapes = shapes.add(weighted_components.sum(1));

    return shapes;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shape_func_forward, "ShapeFunction forward");
}
