

# Partitioning Strategy (equal layers) -- the place indices of isolated modules
iso_module_loc = {
    'resnet32': {
        1: [],  # End-to-end
        2: [[2, 1]],
        3: [[1, 4], [2, 4]],
        4: [[1, 2], [2, 1], [3, 0]],
        8: [[1, 0], [1, 2], [1, 4],
            [2, 1], [2, 3],
            [3, 0], [3, 2]],
        16: [[0, 0],
             [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
             [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
             [3, 0], [3, 1], [3, 2], [3, 3]],
    },
    'resnet110': {
        1: [],  # End-to-end
        2: [[2, 8]],
        3: [[1, 17], [2, 17]],
        4: [[1, 11], [2, 7], [3, 3]],
        8: [[1, 4], [1, 11],
            [2, 0], [2, 7], [2, 14],
            [3, 3], [3, 10]],
        16: [[1, 1], [1, 4], [1, 7], [1, 10], [1, 13], [1, 16],
             [2, 1], [2, 4], [2, 7], [2, 11], [2, 15],
             [3, 1], [3, 5], [3, 9], [3, 13]],
    }
}


# Partitioning Strategy (memory balance) -- the place indices of isolated modules
iso_module_loc_for_memory_balance = {
    'resnet110': {
        'cifar10': {
            1: [],  # End-to-End
            2: [[1, 14]],
            3: [[1, 9], [2, 2]],
            4: [[1, 6], [1, 14], [2, 8]],
        },
        'stl10': {
            1: [],  # End-to-end
            2: [[1, 14]],
            3: [[1, 8], [2, 1]],
            4: [[1, 6], [1, 14], [2, 8]],
        }

    },
    'resnet32': {
        'cifar10': {
            1: [],
            2: [[1, 3]],
            4: [[1, 2], [1, 4], [2, 2]],
            8: [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [3, 1], [3, 2]],
            16: [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
             [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
             [3, 0], [3, 1], [3, 2], [3, 3]]
        }
    }
}


# Scaling factors (ixx, ixy) for losses of local objectives -- local decoder and local classifier
local_loss_scale = {
    'resnet32': {
        'cifar10': {
            2: [5., 1., 0., 0.],
            4: [6., 0., 1., 2.],
            8: [5., 0., 0., 2.],
            16: [5., 0.05, 0.2, 0.5],
        },
    },
    'resnet110': {
        'cifar10': {
            2: [5., 0.5, 0., 0.],
            4: [1., 0., 0., 1.],
            8: [5., 0., 0., 1.],
            16: [5., 0., 0.5, 1.],
        },
        'stl10': {
            2: [10., 1., 0., 0.],
            4: [20., 0.2, 5., 2.],
            8: [20., 0.2, 5., 1.],
            16: [20., 0.5, 10., 0.1],
        },
        'svhn': {
            2: [1., 0.1, 0., 0.],
            4: [5., 0., 0.5, 0.],
            8: [5., 0., 0.5, 2.],
            16: [5., 0.2, 0.5, 1.],
        },
    }
}


# Scaling factors (ixx, ixy) for losses of local objectives -- local decoder and local classifier
# version for memory balance
local_loss_scale_for_memory_balance = {
    'resnet32': {
        'cifar10': {
            2: [5., 0.1, 0., 0.],
            3: [5., 0, 5., 0.1],
            4: [5., 0., 0., 0.1],
            8: [5., 0., 0., 2.],
        },

    },

    'resnet110': {
        'cifar10': {
            2: [5., 0.1, 0., 0.],
            3: [5., 0, 5., 0.1],
            4: [5., 0., 0., 0.1],
        },

    }
}


