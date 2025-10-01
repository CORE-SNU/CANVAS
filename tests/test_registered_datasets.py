from canvas import RegisteredDatasets


def test_registered_datasets():
    dataset = RegisteredDatasets['eth']
    print('dataset dt:', dataset.dt)

    data_array = dataset.asarray()
    print('dataset shape:', data_array.shape)

    scene = dataset.get_scene(timestep=5, history_length=8)
    print(scene)


if __name__ == '__main__':
    test_registered_datasets()