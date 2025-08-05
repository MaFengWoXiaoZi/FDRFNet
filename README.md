<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FDRFNet Project Web App</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom font for a cleaner look */
        body {
            font-family: "Inter", sans-serif;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
    <div class="bg-white p-8 rounded-lg shadow-xl max-w-3xl w-full">
        <!-- Main Title -->
        <h1 class="text-3xl sm:text-4xl font-extrabold text-gray-900 mb-4 text-center leading-tight">
            FDRFNet: A Feature Decoupling and Residual Fusion Network for Multimodal Medical Image Segmentation
        </h1>
        <p class="text-gray-700 text-base sm:text-lg mb-8 text-center">
            This repository contains the code for our paper, "FDRFNet: A Feature Decoupling and Residual Fusion Network for Multimodal Medical Image Segmentation," submitted to the **BIBM 2025** conference.
        </p>

        <hr class="my-8 border-gray-300 rounded-full">

        <!-- Data Download Section -->
        <section class="mb-8">
            <h2 class="text-2xl font-bold text-gray-800 mb-4">Data Download</h2>
            <p class="text-gray-700 mb-4">
                To run the experiments and reproduce our results, please download the following datasets from their respective links:
            </p>
            <ul class="list-disc list-inside space-y-2 text-blue-600">
                <li>
                    <a href="https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2" target="_blank" class="hover:underline font-medium">
                        BraTS: MSD BraTS
                    </a>
                </li>
                <li>
                    <a href="https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2" target="_blank" class="hover:underline font-medium">
                        Prostate: MSD Prostate
                    </a>
                </li>
                <li>
                    <a href="https://zmiclab.github.io/zxh/0/mmwhs/" target="_blank" class="hover:underline font-medium">
                        MMWHS: Multi-Modality Whole Heart Segmentation Challenge
                    </a>
                </li>
            </ul>
        </section>

        <hr class="my-8 border-gray-300 rounded-full">

        <!-- Data Preprocessing, Training, and Inference Section -->
        <section class="mb-8">
            <h2 class="text-2xl font-bold text-gray-800 mb-4">Data Preprocessing, Training, and Inference</h2>
            <p class="text-gray-700 mb-4">
                We recommend using the methodology outlined by <strong class="text-gray-900">nnUNet</strong> for data preprocessing, model training, and testing. You can find more information and resources on their official page:
            </p>
            <p class="text-blue-600">
                <a href="https://github.com/MIC-DKFZ/nnUNet" target="_blank" class="hover:underline font-medium">
                    nnUNet Official Page: MIC-DKFZ/nnUNet
                </a>
            </p>
            <p class="text-gray-700 mt-4">
                Please refer to the nnUNet documentation for detailed instructions on how to prepare the datasets and manage the training/testing pipelines.
            </p>
        </section>

        <hr class="my-8 border-gray-300 rounded-full">

        <!-- Environment Configuration Section -->
        <section>
            <h2 class="text-2xl font-bold text-gray-800 mb-4">Environment Configuration</h2>
            <p class="text-gray-700">
                The complete environment configuration details will be made publicly available upon the acceptance of our paper. We appreciate your understanding.
            </p>
        </section>
    </div>
</body>
</html>
