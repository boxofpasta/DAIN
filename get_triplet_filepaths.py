import glob
import os


def get_triplet_filepaths(input_dir, output_dir):
    """
    :param input_dir: Dataset directory. Each video sequence of images is assumed to be contained within its own folder.
    :param output_dir: The output file path root.
    :return: A list of triplets of (input_a_path, input_c_path, output_b_path).
             The network should predict using images read in from input paths, and output to output_b_path.
             output_b_path will mirror input_b_path, i.e the input_dir substring is replaced with output_dir.
    """
    all_subdirs = get_subdirs(input_dir)

    # Go through each subdirectory and form triplets of (input_a_path, input_c_path, output_b_path).
    # output_b_path will mirror input_b_path, i.e the input_dir substring is replaced with output_dir.
    all_triplets = []
    all_subdirs.sort()
    for subdir in all_subdirs:
        png_files = glob.glob(os.path.join(subdir, '*.png'))
        jpg_files = glob.glob(os.path.join(subdir, '*.jpg'))
        assert len(png_files) * len(jpg_files) == 0
        file_paths = png_files + jpg_files
        assert len(file_paths) > 0
        file_paths.sort()
        for i in range(len(file_paths) - 2):
            input_a_path = file_paths[i]
            input_b_path = file_paths[i + 1]
            input_c_path = file_paths[i + 2]
            output_b_path = os.path.join(output_dir, os.path.relpath(input_b_path, start=input_dir))
            all_triplets.append((input_a_path, input_c_path, output_b_path))
    return all_triplets


def get_pair_filepaths(input_dir, output_dir):
    """
    :param input_dir: Dataset directory. Each video sequence of images is assumed to be contained within its own folder.
    :param output_dir: The output file path root.
    :return: A list of triplets of (input_a_path, input_c_path, output_b_path).
             The network should predict using images read in from input paths, and output to output_b_path.
             output_b_path will mirror input_b_path, i.e the input_dir substring is replaced with output_dir.
    """
    all_subdirs = get_subdirs(input_dir)

    # Go through each subdirectory and form triplets of (input_a_path, input_c_path, output_b_path).
    # output_b_path will mirror input_b_path, i.e the input_dir substring is replaced with output_dir.
    all_triplets = []
    all_subdirs.sort()
    for subdir in all_subdirs:
        png_files = glob.glob(os.path.join(subdir, '*.png'))
        jpg_files = glob.glob(os.path.join(subdir, '*.jpg'))
        assert len(png_files) * len(jpg_files) == 0
        file_paths = png_files + jpg_files
        assert len(file_paths) > 0
        file_paths.sort()
        for i in range(len(file_paths) - 1):
            input_a_path = file_paths[i]
            input_c_path = file_paths[i + 1]
            output_b_path = os.path.join(output_dir, os.path.relpath(input_a_path, start=input_dir))
            all_triplets.append((input_a_path, input_c_path, output_b_path))
    return all_triplets


def get_subdirs(input_dir):
    # Grab all the subdirectories. Each directory is assumed to contain a single video sequence.
    print('Globbing...')
    all_png_files = glob.glob(os.path.join(input_dir, '**', '*.png'), recursive=True)
    all_jpg_files = glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True)
    all_file_paths = all_png_files + all_jpg_files
    all_subdirs_set = set()
    for path in all_file_paths:
        subdir = os.path.dirname(path)
        all_subdirs_set.add(subdir)
    all_subdirs = list(all_subdirs_set)
    print('There are %d subdirectories.' % len(all_subdirs))
    return all_subdirs
