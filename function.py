from init import *


def preprocessing(path, no):
    gray = cv2.imread(path % no, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None
    gray = cv2.resize(gray, (W, H))
    gray = gray.reshape(-1, 1)
    return gray


def imageset(data):
    images = np.empty((H * W, 0))
    for no in range(data.num):
        gray = preprocessing(data.path, no)
        if gray is None:
            raise Exception("영상 파일 읽기 에러")
        images = np.append(images, gray, axis=1)
    return images


def average_image(data):
    sum_images = np.sum(data.set, axis=1, keepdims=True)
    avg_image = sum_images / data.num
    return avg_image


def covariance_matrix(data, average, eigen):
    eig_img = data.set - average
    cov_mat = eig_img.T @ eig_img
    eigen.images = eig_img
    return cov_mat


def get_eigen(covariance):
    eigen_value, eigen_vector = np.linalg.eig(covariance)
    index = np.argsort(eigen_value)[::-1]
    srt_eig_val = eigen_value[index]
    srt_eig_vec = eigen_vector[:, index]
    eigen_sum = np.sum(srt_eig_val) * K
    sorted_sum = 0.0
    for no in range(train.num):
        sorted_sum = sorted_sum + srt_eig_val[no]
        if sorted_sum >= eigen_sum:
            value_index = no
            break
    return srt_eig_val, srt_eig_vec, value_index


def translation_matrix(eigen):
    translation = eigen.images @ eigen.vectors
    return translation


def feature_values(data, average, translation):
    eigen_images = data.set - average
    feats = eigen_images.T @ translation
    return feats


def euclidean_distance(m1, m2):
    dis = sum((m1 - m2)**2)
    return dis


def find_image(in_no):
    min_gap = maxint
    out_no = 0
    for no in range(train.num):
        euc_dis = euclidean_distance(test.feats[in_no], train.feats[no])
        if min_gap > euc_dis:
            min_gap = euc_dis
            out_no = no
    in_image = cv2.imread(test.path % in_no)
    in_image = cv2.resize(in_image, (W * S, H * S))
    in_image = cv2.putText(in_image, "test %03d.jpg" % in_no, t_pos, font, 1, green, 2)
    out_image = cv2.imread(train.path % out_no)
    out_image = cv2.resize(out_image, (W * S, H * S))
    out_image = cv2.putText(out_image, "train %03d.jpg" % out_no, t_pos, font, 1, green, 2)
    return in_image, out_image


def result(no):
    in_img, out_img = find_image(no)
    result_image = cv2.hconcat([in_img, out_img])
    cv2.imshow("result", result_image)
    cv2.moveWindow('result', 100, 100)
    cv2.waitKey(0)


def command_input():
    while True:
        print("-" * 26)
        print("\n\t테스트 영상 번호 입력")
        print("\t- 입력 [ 0 ~ 92 ]")
        print("\t- 종료 [ -1 ]\n")
        print("-" * 26)
        test_no = input("입력 : ")
        print("-" * 26)
        if test_no.isdigit():
            test_no = int(test_no)
            if -1 < test_no < test.num:
                result(test_no)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            else:
                print("입력 범위 초과")
        elif test_no == "-1":
            print("\n종료...")
            break
        else:
            print("잘못된 입력")

