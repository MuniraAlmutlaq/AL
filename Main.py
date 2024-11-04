from ibm_watsonx_ai.foundation_models import Model
import os

class ALLaM_Recognizer:
    @staticmethod
    def get_credentials():
        return {
            "url": "https://eu-de.ml.cloud.ibm.com",
            "api_key": "RGFpLsHNJvfKImXgyKACDUWgAmj9A8e62cSLS5cgP17u"
        }

    model_id = "sdaia/allam-1-13b-instruct"
    project_id = "b04555df-6516-4591-9826-2e7bf07168db"

    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 1520,
        "repetition_penalty": 1
    }

    space_id = os.getenv("SPACE_ID")

    model = Model(
        model_id= model_id,
        credentials= get_credentials(),
        params= parameters,
        project_id= project_id,
        space_id= space_id
    )

    @staticmethod
    def process_text(extracted_text):
        #Preparing the prompt for the model.
        prompt_input = """Generate the processed text after applying the following processes at the same time:
        First, spelling correction.
        Second, grammar correction.
        Third, ضع العلامات الإعرابية على النص.
        Lastly, clarity of the context.


        Input:  لا أحب الحياة الجديده ولا مظاهرها ولا بيوتها ولا
        ساكنيها أريد ذالك البيت الصغير والحارة الصغيرة
        التى لا تتسابق فيهاالطوابق تعلو بعضها الآخر أريد
        تللكت الأرجوحة الزرقاء التي كانت وطن لضحكاتنا أريد

        إنتظار العيد بالحناء والغناء أريد تلكالفرحة اللتي

        تستقبل المطر أريدنا نحن متجردون من كل هذه
         التكاليف اريدنا ان نعود إلى الزمان حيث لا جدران تفصلنا ولاأبواب توصدنا
        Output: لَا أَحُبَّ الْحَيَاةِ الْجَدِيدَةِ وَلَا مَظَاهِرُهَا وَلَا بُيُوتُهَا وَلَا سَاكِنِيهَا. أُرِيدُ ذَلِكَ الْبَيْتِ الصَّغِيرِ وَالْحَارَّةِ الصَّغِيرَةِ الَّتِي لَا تَتَسَابَقْ فِيهَا الطَّوَابِقَ تَعْلُو بَعْضُهَا الْآخِرَ. أُرِيدُ تِلْكَ الْأُرْجُوحَةِ الزَّرْقَاءِ الَّتِي كانت وَطَنٌ لِضَحْكَاتِنَا. أُرِيدُ إنتظار الْعِيدَ بِالْحِنَّاءِ وَالْغِنَاءِ. أُرِيدُ تِلْكَ الْفَرَحَةِ الَّتِي تَسْتَقْبِلُ الْمَطَرُ. أُرِيدُنَا نَحْنُ مُتَجَرِّدُونَ مِنْ كُلُّ هَذِهِ التَّكَاليفَ. أُرِيدُنَا أَنَّ نَعُودُ إِلَى الزَّمَانِ حَيْثُ لَا جُدْرَانُ تَفَصُّلِنَا ولا أبواب تُوصِدُنَا.

        Input: السلام عليكم ورحمة الله وبركاته
        Output: السُّلَّامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ وَبَرَكَاتُهُ.

        Input: قام الملك عبد العزيز بعمل بطولي من أجل استرداد ملك أبائه وأجداده الذي سلبه الأعداء من الخارج والداخل؛ ونجح في استرداد
        مدينة الرياض عاصمة الدولة السعودية الثانية في قلة قليلة من العدة والعتاد.
        Output: قَامَ الْمَلِكُ عَبْدُ الْعَزِيزِ بِعَمَلٍ بَطُولِيٍّ مِنْ أَجْلِ اسْتِرْدَادِ مَمْلَكَةِ آبَائِهِ وَأَجْدَادِهِ الَّتِي سَلَبَهَا الْأَعْدَاءُ مِنَ الْخَارِجِ وَالْدَّاخِلِ؛ وَنَجَحَ فِي اسْتِرْدَادِ مَدِينَةِ الرَّيَاضِ عَاصِمَةِ الدَّوْلَةِ السَّعُودِيَّةِ الثَّانِيَةِ فِي قِلَّةِ قَلِيلَةٍ مِنَ الْعُدَّةِ وَالْعَتَادِ.

        Input: وسطّر في التاريخ أروع مثل في التضحية من أجل الوطن
        وترابه مُعرّضاً نفسه للمخاطر التي كانت تتربص به من قبل أعدائه؛ وتمكن بعد ذلك من استكمال مشروعه الوطني في توحيد واخضاع
        مناطق شبه الجزيرة العربية (نجد- القصيم- الإحساء- جبل شمر عسير الحجاز- المخلاف السليماني «جازان») والتي دانت له جميعها
        بالسمع والطاعة.
        Output: وَسَطَّرَ فِي التَّارِيخِ أَرْوَعْ مَثَلٍ فِي التَّضْحِيَةِ مِنْ أَجْلِ الْوَطَنِ وَتُرَابِهِ مُعَرِّضًا نَفْسَهُ لِلْمَخَاطِرِ الَّتِي كَانَتْ تَتَرَبَّصُ بِهِ مِنْ قِبْلِ أَعْدَائِهِ؛ وَمَكَّنَ بَعْدَ ذَلِكَ مِنْ اسْتِكْمَالِ مَشْرُوعِهِ الْوَطَنِيِّ فِي تَوْحِيدِ وَإخْضَاعِ مَنَاطِقِ شِبْهِ الْجَزِيرَةِ الْعَرَبِيَّةِ (نَجْدِ- الْقَصِيمِ- الْإِحْسَاءِ- جَبَلِ شَمَّرَ- عَسِيرَ- الْحِجَازِ- الْمَخْلَافِ السُّلَيْمَانِيِّ «جَازَانَ») وَالَّتِي دَانَتْ لَهُ جَمِيعُهَا
        بِالْسَمْعِ وَالطَاعَةِ.

        Input: 1797ه/1611م
        Output: 1797هـ/1611م

        Input: عمل علي بناء
        Output: عَمِلَ عَلَى بِنَاءِ.

        Input:  إال أنهــا تركــت يي البالد النجديــة
        Output: إِلَّا أَنَّهَا تَرَكَتْ في البلاد النَّجْدِيَّةِ.

        Input: نمذاج علام ااالمهتص يي النصوص العربيه
        Output: نَمُوذَجُ عَلَّام المختص فِي النَّصُوصِ الْعَرَبِيَّةِ.

        Input: فرثق منييره ومهند يي ت حدي علام
        Output: فَرِيقُ مُنِيرَه وَمُهَنَّدٍ فِي تَحُدِّي عُلَّامَ.

        Input: ااالذي
        Output: الَّذِي

        Input: مااااههر
        Output: مَاهِرٌ

        Input: سييتي
        Output: سَيَأْتِي

        Input: السيياااح  ه
        Output: السِّيَاحَةُ"""
        formatted_question= f"<s> [INST] {extracted_text} [\\INST]"
        prompt= f"{prompt_input} {formatted_question}"

        #Generating the response from the model.
        generated_response= "Output: "
        generated_response+= ALLaM_Recognizer.model.generate_text(prompt= prompt, guardrails= False)

        lines = generated_response.splitlines()
        result = [line for line in lines if line.startswith("Output:")]

        if result:
            extracted = result[0].split("Output: ")[1].strip()
            
        return extracted

    @staticmethod
    def correct_arabic_text_file(input_file, output_file):
        #Read the Arabic text from the input file.
        with open(input_file, "r", encoding= "utf-8") as file:
            extracted_text = file.read()

        #Process the text.
        corrected_text = ALLaM_Recognizer.process_text(extracted_text)

        #Write the corrected text to the output file.
        with open(output_file, "w", encoding= "utf-8") as file:
            file.write(corrected_text)

        print(f"Corrected text has been written to {output_file}")  #Confirmation message.


if __name__ == "__main__":
    input_file= "Website/TemporaryDatabase/arabic_text.txt"  #Path to Reading the Arabic text from arabic_text.txt document.
    output_file= "Website/TemporaryDatabase/corrected_text.txt"  #Path to save the corrected text in corrected_text.txt document.

    ALLaM_Recognizer.correct_arabic_text_file(input_file, output_file)
