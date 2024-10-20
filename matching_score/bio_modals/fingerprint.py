### built-in modules

### 3rd-party modules
import numpy as np
import cv2

### project modules
from bio_modals.neurotecbase import *


class Fingerprint(NeurotecBase):
    def __init__(self, license_path=''):
        NeurotecBase.__init__(self, license_path)
        self.check_license('Biometrics.FingerExtraction,Biometrics.FingerMatching')

    def extract_feature(self, img_or_subject):
        img_width = 0
        img_height = 0
        minutia_set = []

        if type(img_or_subject) == self.SDK.Biometrics.NSubject:  ## subject로 입력되었을 경우
            subject = img_or_subject

        else:
            subject, quality = self.create_subject(img_or_subject)  ## image로 입력되었을 경우
            if subject is None:
                return subject, quality, minutia_set

        # NFRecord_to_array
        template_buffer = subject.GetTemplateBuffer().ToArray()  # template 고치기
        template = self.SDK.Biometrics.NTemplate(template_buffer)  #
        count = 0
        for nfRec in template.Fingers.Records:
            img_quality = nfRec.Quality
            minutiaFormat = nfRec.MinutiaFormat
            index = 0
            for minutia in nfRec.Minutiae:
                count += 1
                x = minutia.X
                y = minutia.Y

                direction = (2.0 * minutia.RawAngle * 360.0 + 256.0) / (2.0 * 256.0)  ## 256?확인
                direction -= 90
                if ((
                        minutiaFormat & self.SDK.Biometrics.NFMinutiaFormat.HasQuality) == self.SDK.Biometrics.NFMinutiaFormat.HasQuality):
                    quality = minutia.Quality
                else:
                    continue
                # 1 - ending, 2 - bifurcation
                minutia_Type = (minutia.Type.value__ * 128) - 1

                minutia_set.append([x, y, direction, quality, minutia_Type])
        print(subject, img_quality, minutia_set)
        return subject, img_quality, minutia_set, count

    def make_condition_image(self, feature_vector, position_angle_change: Optional[list] = None):
        pass

    def make_pair_image(self, image):
        pass

    def create_subject(self, img_or_file):
        subject = self.SDK.Biometrics.NSubject()
        finger = self.SDK.Biometrics.NFinger()
        try:
            if type(img_or_file) == str:
                nimage = self.SDK.Images.NImage.FromFile(img_or_file)
            elif type(img_or_file) == np.ndarray:
                ww, hh = img_or_file.shape[1::-1]
                cc = 1
                if len(img_or_file.shape) == 3:
                    cc = img_or_file.shape[2]
                pixelformat = self.SDK.Images.NPixelFormat.Rgb8U if cc == 3 else self.SDK.Images.NPixelFormat.Grayscale8U
                nimage = self.SDK.Images.NImage.FromData(pixelformat, ww, hh, 0, ww * cc, self.SDK.IO.NBuffer.FromArray(img_or_file.tobytes()))
            else:
                raise Exception
        except:
            raise TypeError('type is not supported')

        ''' code from Binh
        nimage.ResolutionIsAspectRatio = False
        biometricClient.FingersTemplateSize = NTemplateSize.Small
        '''
        nimage.HorzResolution = 500
        nimage.VertResolution = 500

        finger.Image = nimage
        subject.Fingers.Add(finger)

        if self.biometricClient.CreateTemplate(subject) != self.SDK.Biometrics.NBiometricStatus.Ok:
            return None, None
        quality = subject.GetTemplate().Fingers.Records.get_Item(0).Quality
        return subject, quality


def unit_test_match(obj):
    print('######################### Unit Test 1 - grayscale image input #########################')
    img1 = cv2.imread("../unit_test_data/Fingerprint/093/L3_03.BMP", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("../unit_test_data/Fingerprint/093/L3_04.BMP", cv2.IMREAD_GRAYSCALE)
    matching_score, quality1, quality2 = obj.match(img1, img2)
    print(matching_score, quality1, quality2)
    print('######################### Unit Test 2 - color image input #########################')
    img1 = cv2.imread("../unit_test_data/Fingerprint/093/L3_03.BMP", cv2.IMREAD_COLOR)
    img2 = cv2.imread("../unit_test_data/Fingerprint/093/L3_04.BMP", cv2.IMREAD_COLOR)
    matching_score, quality1, quality2 = obj.match(img1, img2)
    print(matching_score, quality1, quality2)
    print('######################### Unit Test 3 - file path input #########################')
    img1 = "../unit_test_data/Fingerprint/093/L3_03.BMP"
    img2 = "../unit_test_data/Fingerprint/093/L3_04.BMP"
    matching_score, quality1, quality2 = obj.match(img1, img2)
    print(matching_score, quality1, quality2)
    print('######################### Unit Test 4 - weired path input #########################')
    img1 = r'C:\weired\path'
    img2 = r'C:\weired\path2'
    try:
        matching_score, quality1, quality2 = obj.match(img1, img2)
        print(matching_score, quality1, quality2)
    except Exception as e:
        print(str(e))
    print()


def unit_test_match_using_filelist(obj):
    print('######################### Unit Test 1 - filelist1 #########################')
    filelist1 = [
        "../unit_test_data/Fingerprint/094/R3_01.BMP",
        "../unit_test_data/Fingerprint/094/R3_01.BMP",
        "../unit_test_data/Fingerprint/227/L2_04.BMP",
    ]
    results, qualities1, qualities2 = obj.match_using_filelist(filelist1)
    print(results, qualities1, qualities2)

    print('################### Unit Test 2 - filelist1 and filelist2 ###################')
    filelist2 = [
        "../unit_test_data/Fingerprint/093/L3_04.BMP",
        "../unit_test_data/Fingerprint/094/R3_02.BMP",
        "../unit_test_data/Fingerprint/227/L2_05.BMP",
    ]
    results, qualities1, qualities2 = obj.match_using_filelist(filelist1, filelist2)
    print(results, qualities1, qualities2)

    print('######################### Unit Test 3 - file error #########################')
    filelist3 = [r"C:\weired\path", r"C:\weired\path2"]
    try:
        results, qualities1, qualities2 = obj.match_using_filelist(filelist3)
        print(results, qualities1, qualities2)
    except Exception as e:
        print(str(e))
    print()


if __name__ == '__main__':
    obj = Fingerprint(r"C:\Users\CVlab\Desktop\Neurotec_Biometric_12_4_SDK_2023-05-17\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64")

    unit_test_match(obj)
    unit_test_match_using_filelist(obj)