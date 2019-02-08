// author: Justus Schock (justus.schock@rwth-aachen.de)

#include <torch/torch.h>
#include <vector>
#include <iostream>

at::Tensor _ensemble_2d_matrix(at::Tensor homogen_matrix,
                               at::Tensor rotation_params,
                               at::Tensor translation_params,
                               at::Tensor scale_params){

    at::IntList size_list = {scale_params.size(0), 1, 1};

    rotation_params = rotation_params.squeeze(-1);
    scale_params = scale_params.squeeze(-1);
    translation_params = translation_params.squeeze(-1);

    at::Tensor trafo_matrix = homogen_matrix.repeat(size_list).clone();

    // matrix[:, 0, 0] = s*cos\theta
    trafo_matrix.narrow(1, 0, 1).narrow(2, 0, 1) = (scale_params * rotation_params.cos()).narrow(1, 0, 1).clone();

    // matrix[:, 0, 1} = s*sin\theta
    trafo_matrix.narrow(1, 0, 1).narrow(2, 1, 1) = (scale_params * rotation_params.sin()).narrow(1, 0, 1).clone();

    // matrix[:, 1, 0] = -s*sin\theta
    trafo_matrix.narrow(1, 1, 1).narrow(2, 0, 1) = (-scale_params * rotation_params.sin()).narrow(1, 0, 1).clone();

    // matrix[:, 1, 1] = s*cos\theta
    trafo_matrix.narrow(1, 1, 1).narrow(2, 1, 1) = (scale_params * rotation_params.cos()).narrow(1, 0, 1).clone();

    trafo_matrix.narrow(1, 0, trafo_matrix.size(1)-1).narrow(2, trafo_matrix.size(2)-1, 1) = translation_params.clone();

    return trafo_matrix;
}

at::Tensor _ensemble_3d_matrix(at::Tensor homogen_matrix,
                               at::Tensor rotation_params,
                               at::Tensor translation_params,
                               at::Tensor scale_params){

    at::IntList size_list = {scale_params.size(0), 1, 1};

    at::Tensor trafo_matrix = homogen_matrix.repeat(size_list).clone();

    rotation_params = rotation_params.squeeze(-1);
    scale_params = scale_params.squeeze(-1);
    translation_params = translation_params.squeeze(-1);

    auto roll = rotation_params.narrow(1, 2, 1);
    auto pitch = rotation_params.narrow(1, 1, 1);
    auto yaw = rotation_params.narrow(1, 0, 1);


    // matrix{:, 0, 0] = s*(cos(pitch)*cos(roll))
    trafo_matrix.narrow(1, 0, 1).narrow(2, 0, 1) = (scale_params * (pitch.cos() * roll.cos())).narrow(1, 0, 1).clone();

    // matrix[:, 0, 1] = s*(cos(pitch)*sin(roll))
    trafo_matrix.narrow(1, 0, 1).narrow(2, 1, 1) = (scale_params * (pitch.cos() * roll.sin())).narrow(1, 0, 1).clone();

    // matrix[:, 0, 2] = s*(-sin(pitch))
    trafo_matrix.narrow(1, 0, 1).narrow(2, 2, 1) = (scale_params * (-pitch.sin())).narrow(1, 0, 1).clone();

    // matrix[:, 1, 0] = s*(sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll))
    trafo_matrix.narrow(1, 1, 1).narrow(2, 0, 1) = (scale_params * (yaw.sin() * pitch.sin() * roll.cos() - yaw.cos() * roll.sin())).narrow(1, 0, 1).clone();

    // matrix[:, 1, 1] = s*(sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll))
    trafo_matrix.narrow(1, 1, 1).narrow(2, 1, 1) = (scale_params * (yaw.sin() * pitch.sin() * roll.sin() + yaw.cos() * roll.cos())).narrow(1, 0, 1).clone();

    // matrix[:, 1, 2] = s*(sin(yaw)*cos(pitch))
    trafo_matrix.narrow(1, 1, 1).narrow(2, 2, 1) = (scale_params * (yaw.sin() * pitch.cos())).narrow(1, 0, 1).clone();

    // matrix[:, 2, 0] = s*(cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll))
    trafo_matrix.narrow(1, 2, 1).narrow(2, 0, 1) = (scale_params * (yaw.cos() * pitch.sin() * roll.cos() + yaw.sin() * roll.sin())).narrow(1, 0, 1).clone();

    // matrix[:, 2, 1] = s*(cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll))
    trafo_matrix.narrow(1, 2, 1).narrow(2, 1, 1) = (scale_params * (yaw.cos() * pitch.sin() * roll.sin() - yaw.sin() * roll.cos())).narrow(1, 0, 1).clone();

    // matrix[:, 2, 1] = s*(cos(yaw)*cos(pitch))
    trafo_matrix.narrow(1, 2, 1).narrow(2, 2, 1) = (scale_params * (yaw.cos() * pitch.cos())).narrow(1, 0, 1).clone();

    // translation params
    trafo_matrix.narrow(1, 0, trafo_matrix.size(1)-1).narrow(2, trafo_matrix.size(2)-1, 1) = translation_params.clone();

    return trafo_matrix;

}

at::Tensor _ensemble_trafo(at::Tensor homogen_matrix,
                           at::Tensor rotation_params,
                           at::Tensor translation_params,
                           at::Tensor scale_params){
    if (homogen_matrix.size(1) == 3) {
        return _ensemble_2d_matrix(homogen_matrix,
                                   rotation_params,
                                   translation_params,
                                   scale_params);
    }
    else if (homogen_matrix.size(1) == 4){
        return _ensemble_3d_matrix(homogen_matrix,
                                   rotation_params,
                                   translation_params,
                                   scale_params);
    }
}

at::Tensor forward_homogeneous_layer(at::Tensor shapes,
                                     at::Tensor homogen_matrix,
                                     at::Tensor rotation_params,
                                     at::Tensor translation_params,
                                     at::Tensor scale_params){

    auto trafo_matrix = _ensemble_trafo(homogen_matrix, rotation_params, translation_params, scale_params);

    auto homogen_shapes = at::cat({shapes, at::ones({shapes.size(0), shapes.size(1), 1}, shapes.options())}, -1);
    
    auto transformed_shapes = at::bmm(homogen_shapes, trafo_matrix.permute({0, 2, 1}));

    return transformed_shapes.narrow(-1, 0, homogen_shapes.size(-1) - 1).div(transformed_shapes.narrow(-1, -1, 1).unsqueeze(-1));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_homogeneous_layer, "Homogeneous Transformation forward");
}
