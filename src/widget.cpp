// widget.cpp

#include <QFontDatabase>
#include <QImageReader>
#include <QApplication>
#include <QCloseEvent>
#include <QMessageBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QImage>
#include <QTime>
#include <QTimer>
#include <QFont>
#include <QDir>

#include <algorithm>
#include <string>
#include <vector>
#include <cmath>

#include "widget.h"
#include "ui_widget.h"
#include "mesh.h"
#include "opencl_interface.h"

Widget::Widget(QWidget *parent):
    QWidget(parent),
    t(0.0f),
    step(0.1f),
    constSpeed(false),
    sampleNum(0),
    isMinMaxAvailable(false),
    isColorMapLoaded(false),
    isNormalMapLoaded(false),
    renderer(clInterface),
    ui(new Ui::Widget) {

    // Add Font
    QFontDatabase::addApplicationFont(":/font/JetBrainsMono-Regular.ttf");
    ui->setupUi(this);

    // Initialize UI
    ui->select3DModelPushButton->setText("Select (.obj, .ply, .stl)");
    ui->samplerCheckBox->setChecked(false);
    ui->samplerComboBox->setEnabled(false);
    ui->colorMapCheckBox->setChecked(false);
    ui->selectColorMapPushButton->setText("Select (.png, .jpg, ...)");
    ui->selectColorMapPushButton->setEnabled(false);
    ui->normalMapCheckBox->setChecked(false);
    ui->selectNormalMapPushButton->setText("Select (.png, .jpg, ...)");
    ui->selectNormalMapPushButton->setEnabled(false);
    ui->generateGrayscalePushButton->setEnabled(false);
    ui->generateGeopotentialPushButton->setEnabled(false);
    ui->statusText->setText("Please select a 3D model.");
    setProgressBar(true, 0);

    // Disable Deformity Calculator
    disableDeformityCalculator();

    // Get GPU List
    std::vector<std::string> deviceList = clInterface.getDeviceList();
    ui->clDeviceComboBox->clear();
    deviceCount = 0;
    for (std::string device : deviceList) {
        ui->clDeviceComboBox->addItem(QString::fromStdString(device));
        deviceCount++;
    }
    if (deviceCount != 0) {
        ui->clDeviceComboBox->setCurrentIndex(0);
    }

    // Connect Signals and Slots
    QObject::connect(ui->select3DModelPushButton, &QPushButton::clicked, this, &Widget::onSelect3DModelPushButtonClicked);
    QObject::connect(ui->imageFormatComboBox, &QComboBox::currentTextChanged, this, &Widget::onImageFormatChanged);
    QObject::connect(ui->samplerCheckBox, &QCheckBox::clicked, this, &Widget::onSamplerCheckBoxClicked);
    QObject::connect(ui->samplerComboBox, &QComboBox::currentIndexChanged, this, &Widget::onSamplerChanged);
    QObject::connect(ui->colorMapCheckBox, &QCheckBox::clicked, this, &Widget::onColorMapCheckBoxClicked);
    QObject::connect(ui->selectColorMapPushButton, &PushButton::clicked, this, &Widget::onSelectColorMapPushButtonClicked);
    QObject::connect(ui->normalMapCheckBox, &QCheckBox::clicked, this, &Widget::onNormalMapCheckBoxClicked);
    QObject::connect(ui->selectNormalMapPushButton, &PushButton::clicked, this, &Widget::onSelectNormalMapPushButtonClicked);
    QObject::connect(ui->generateGrayscalePushButton, &QPushButton::clicked, this, &Widget::onGenerateGrayscalePushButtonClicked);
    QObject::connect(ui->generateGeopotentialPushButton, &QPushButton::clicked, this, &Widget::onGenerateGeopotentialPushButtonClicked);
    QObject::connect(ui->realRadiusDoubleSpinBox, &QDoubleSpinBox::valueChanged, this, &Widget::onRealRadiusValueChanged);
    QObject::connect(&fakeProgressTimer, &QTimer::timeout, this, &Widget::updateFakeProgress);
}

Widget::~Widget() {
    delete ui;
}

void Widget::closeEvent(QCloseEvent *event) {
    if (meshDispatcher.isRunning() ||
        bvhDispatcher.isRunning() ||
        renderer.isRunning() ||
        imageSaver.isRunning() ||
        geopotentialDispatcher.isRunning()) {

        exit(-1);
    }
    else {
        event->accept();
    }
}

void Widget::disableUI() {
    ui->select3DModelPushButton->setEnabled(false);
    ui->resolutionSpinBox_x->setEnabled(false);
    ui->resolutionSpinBox_y->setEnabled(false);
    ui->clDeviceComboBox->setEnabled(false);
    ui->imageFormatComboBox->setEnabled(false);
    ui->samplerCheckBox->setEnabled(false);
    ui->samplerComboBox->setEnabled(false);
    ui->grayscaleDepthComboBox->setEnabled(false);
    ui->xFlipCheckBox->setEnabled(false);
    ui->yFlipCheckBox->setEnabled(false);
    ui->primeMeridianDoubleSpinBox->setEnabled(false);
    ui->colorMapCheckBox->setEnabled(false);
    ui->selectColorMapPushButton->setEnabled(false);
    ui->normalMapCheckBox->setEnabled(false);
    ui->selectNormalMapPushButton->setEnabled(false);
    ui->outputTXTCheckBox->setEnabled(false);
    ui->generateGrayscalePushButton->setEnabled(false);
    ui->generateGeopotentialPushButton->setEnabled(false);

    disableDeformityCalculator();
}

void Widget::enableUI() {
    ui->select3DModelPushButton->setEnabled(true);
    ui->resolutionSpinBox_x->setEnabled(true);
    ui->resolutionSpinBox_y->setEnabled(true);
    ui->clDeviceComboBox->setEnabled(true);
    ui->imageFormatComboBox->setEnabled(true);
    ui->samplerCheckBox->setEnabled(true);
    ui->samplerComboBox->setEnabled(ui->samplerCheckBox->isChecked());
    ui->grayscaleDepthComboBox->setEnabled(true);
    ui->xFlipCheckBox->setEnabled(true);
    ui->yFlipCheckBox->setEnabled(true);
    ui->primeMeridianDoubleSpinBox->setEnabled(true);
    ui->colorMapCheckBox->setEnabled(true);
    ui->selectColorMapPushButton->setEnabled(ui->colorMapCheckBox->isChecked());
    ui->normalMapCheckBox->setEnabled(true);
    ui->selectNormalMapPushButton->setEnabled(ui->normalMapCheckBox->isChecked());
    ui->outputTXTCheckBox->setEnabled(true);

    if (deviceCount != 0) {
        ui->generateGrayscalePushButton->setEnabled(mesh.isMeshLoaded());
    }
    ui->generateGeopotentialPushButton->setEnabled(mesh.isMeshLoaded());

    if (isMinMaxAvailable) {
        enableDeformityCalculator();
    }
    else {
        disableDeformityCalculator();
    }
}

void Widget::setProgressBar(bool enable, int percent) {
    if (enable) {
        ui->progressBar->setMinimum(0);
        ui->progressBar->setMaximum(100);
        ui->progressBar->setValue(percent);
    }
    else {
        ui->progressBar->setMinimum(0);
        ui->progressBar->setMaximum(0);
    }
}

void Widget::disableDeformityCalculator() {
    ui->minDistanceText->setText("N/A");
    ui->maxDistanceText->setText("N/A");
    ui->deformityText->setText("N/A");
    ui->realRadiusDoubleSpinBox->setEnabled(false);
    ui->realDeformityText->setText("N/A");
    ui->scalingFactorText->setText("N/A");
}

void Widget::enableDeformityCalculator() {
    if (isMinMaxAvailable) {
        double realRadius = ui->realRadiusDoubleSpinBox->value();
        double scalingFactor = realRadius / min;
        double realDeformity = max * scalingFactor - realRadius;

        ui->minDistanceText->setText(QString::number(min, 'f', 8));
        ui->maxDistanceText->setText(QString::number(max, 'f', 8));
        ui->deformityText->setText(QString::number(max - min, 'f', 8));
        ui->realRadiusDoubleSpinBox->setEnabled(true);
        ui->realDeformityText->setText(QString::number(realDeformity, 'f', 8));
        ui->scalingFactorText->setText(QString::number(scalingFactor, 'f', 8));
    }
}

void Widget::errorMessageBox(const QString &title, const QString &text) {
    QMessageBox msgBox;
    QFont font("JetBrains Mono");
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.setWindowTitle(title);
    msgBox.setText(text);
    msgBox.setFont(font);
    msgBox.exec();
}

void Widget::onSelect3DModelPushButtonClicked() {
    QString filePath = QFileDialog::getOpenFileName(this,
                                                    "Select 3D Model",
                                                    QDir::currentPath(),
                                                    "All files (*.*);;OBJ (*.obj);;PLY (*.ply);;STL (*.stl)");
    if (filePath.isEmpty()) {
        return;
    }
    QFileInfo fileInfo(filePath);
    tempMeshFileName = fileInfo.fileName();
    tempMeshFilePath = filePath;
    if (fileInfo.exists()) {
        // Save Old Text
        oldSelectButtonText = ui->select3DModelPushButton->text();
        oldStatus = ui->statusText->text();
        oldVerticesText = ui->verticesText->text();
        oldFacesText = ui->facesText->text();

        // Disable UI
        disableUI();
        disableDeformityCalculator();
        ui->select3DModelPushButton->setText("Loading...");
        ui->verticesText->setText("Loading...");
        ui->facesText->setText("Loading...");

        // Set Status Text
        QString statusTextText = "Loading " + filePath;
        if (statusTextText.length() > 47) {
            statusTextText = statusTextText.left(47) + "...";
        }
        else {
            statusTextText += "...";
        }
        ui->statusText->setText(statusTextText);

        // Reset Progress Bar
        fakeProgressTimer.stop();
        setProgressBar(true, 100);

        t = 0.0f;
        step = 0.15f;
        constSpeed = false;

        fakeProgressTimer.start(50);

        QObject::connect(&meshDispatcher,
                         &MeshDispatcher::finishedLoading,
                         this,
                         &Widget::handleMeshLoadingResult,
                         Qt::UniqueConnection);

        meshDispatcher.setParameter(&mesh, filePath.toStdString());
        meshDispatcher.start();
    }
}

void Widget::onImageFormatChanged(QString currentFormat) {
    if (currentFormat == "JPG") {
        ui->grayscaleDepthComboBox->clear();
        ui->grayscaleDepthComboBox->addItem("8 Bit");
        ui->grayscaleDepthComboBox->setCurrentIndex(0);
    }
    else {
        if (ui->grayscaleDepthComboBox->count() < 2) {
            ui->grayscaleDepthComboBox->clear();
            ui->grayscaleDepthComboBox->addItem("8 Bit");
            ui->grayscaleDepthComboBox->addItem("16 Bit");
            ui->grayscaleDepthComboBox->setCurrentIndex(0);
        }
    }
}

void Widget::onSamplerCheckBoxClicked(bool isChecked) {
    ui->samplerComboBox->setEnabled(isChecked);
    if (isChecked) {
        sampleNum = ui->samplerComboBox->currentIndex() + 1;
    }
    else {
        sampleNum = 0;
    }
}

void Widget::onSamplerChanged(int currentIndex) {
    sampleNum = currentIndex + 1;
}

void Widget::onColorMapCheckBoxClicked(bool isChecked) {
    ui->selectColorMapPushButton->setEnabled(isChecked);
}

void Widget::onSelectColorMapPushButtonClicked() {
    QString filePath = QFileDialog::getOpenFileName(this,
                                                    "Select Color Map",
                                                    QDir::currentPath(),
                                                    "All files (*.*)");
    if (filePath.isEmpty()) {
        return;
    }

    QString oldText = ui->selectColorMapPushButton->text();
    ui->selectColorMapPushButton->setText("Loading...");
    disableUI();
    QApplication::processEvents();

    QFileInfo fileInfo(filePath);
    QString tempColorMapFileName = fileInfo.fileName();
    QString tempColorMapFilePath = filePath;
    if (fileInfo.exists()) {
        QImageReader::setAllocationLimit(4096);
        QImage colorMapImage(tempColorMapFilePath);
        if (colorMapImage.isNull()) {
            errorMessageBox("Error", "Error loading "
                                                 + tempColorMapFilePath
                                                 + ". Image corrupted or too large.");

            ui->selectColorMapPushButton->setText(oldText);
            enableUI();
            return;
        }
        colorMapImage = colorMapImage.convertToFormat(QImage::Format_RGBA8888);
        colorMapWidth = colorMapImage.width();
        colorMapHeight = colorMapImage.height();

        const uchar *imageData = colorMapImage.bits();
        std::size_t imageSize = static_cast<std::size_t>(colorMapImage.sizeInBytes());
        colorMapArray.reset(new unsigned char[imageSize]);
        memcpy(colorMapArray.get(), imageData, imageSize);

        currentColorMapFileName = tempColorMapFileName;
        currentColorMapFilePath = tempColorMapFilePath;
        isColorMapLoaded = true;
        ui->selectColorMapPushButton->setText(currentColorMapFileName);
        enableUI();
    }
}

void Widget::onNormalMapCheckBoxClicked(bool isChecked) {
    ui->selectNormalMapPushButton->setEnabled(isChecked);
}

void Widget::onSelectNormalMapPushButtonClicked() {
    QString filePath = QFileDialog::getOpenFileName(this,
                                                    "Select Normal Map",
                                                    QDir::currentPath(),
                                                    "All files (*.*)");
    if (filePath.isEmpty()) {
        return;
    }

    QString oldText = ui->selectNormalMapPushButton->text();
    ui->selectNormalMapPushButton->setText("Loading...");
    disableUI();
    QApplication::processEvents();

    QFileInfo fileInfo(filePath);
    QString tempNormalMapFileName = fileInfo.fileName();
    QString tempNormalMapFilePath = filePath;
    if (fileInfo.exists()) {
        QImageReader::setAllocationLimit(4096);
        QImage normalMapImage(tempNormalMapFilePath);
        if (normalMapImage.isNull()) {
            errorMessageBox("Error", "Error loading "
                                                 + tempNormalMapFilePath
                                                 + ". Image corrupted or too large.");

            ui->selectNormalMapPushButton->setText(oldText);
            enableUI();
            return;
        }
        normalMapImage = normalMapImage.convertToFormat(QImage::Format_RGBA8888);
        normalMapWidth = normalMapImage.width();
        normalMapHeight = normalMapImage.height();

        const uchar *imageData = normalMapImage.bits();
        std::size_t imageSize = static_cast<std::size_t>(normalMapImage.sizeInBytes());
        normalMapArray.reset(new unsigned char[imageSize]);
        memcpy(normalMapArray.get(), imageData, imageSize);

        currentNormalMapFileName = tempNormalMapFileName;
        currentNormalMapFilePath = tempNormalMapFilePath;
        isNormalMapLoaded = true;
        ui->selectNormalMapPushButton->setText(currentNormalMapFileName);
        enableUI();
    }
}

void Widget::onGenerateGrayscalePushButtonClicked() {
    // State Check
    if (!mesh.isMeshLoaded()) {
        errorMessageBox("Error", "No mesh selected.");
        return;
    }
    if (ui->colorMapCheckBox->isChecked()) {
        if (!isColorMapLoaded) {
            errorMessageBox("Error", "No color map selected.");
            return;
        }
        if (!mesh.hasUVCoord()) {
            errorMessageBox("Error", "Selected 3D model does not contain UV coordinates.");
            return;
        }
    }
    if (ui->normalMapCheckBox->isChecked()) {
        if (!isNormalMapLoaded) {
            errorMessageBox("Error", "No normal map selected.");
            return;
        }
        if (!mesh.hasUVCoord()) {
            errorMessageBox("Error", "Selected 3D model does not contain UV coordinates.");
            return;
        }
    }

    // Disable UI
    disableUI();

    // Update Status
    ui->statusText->setText("[1/2] Building BVH...");

    // Restart Timer
    fakeProgressTimer.stop();
    setProgressBar(true, 0);

    t = 0.0f;
    step = 600000.0f / mesh.faceCount();
    constSpeed = true;

    fakeProgressTimer.start(20);

    QObject::connect(&bvhDispatcher,
                     &BVHDispatcher::finishedBuilding,
                     this,
                     &Widget::startRenderer,
                     Qt::UniqueConnection);

    // Dispatch BVH
    bvhDispatcher.setParameter(&bvh, &mesh);
    bvhDispatcher.start();
}

void Widget::onGenerateGeopotentialPushButtonClicked() {
    // Disable UI
    disableUI();

    // Update Status
    ui->statusText->setText("Calculating Geopotential...");

    // Set Progress Bar
    setProgressBar(false);

    QObject::connect(&geopotentialDispatcher,
                     &GeopotentialDispatcher::finishedCalculating,
                     this,
                     &Widget::handleGeopotentialResult,
                     Qt::UniqueConnection);

    // Get Current Time
    QTime currentTime = QTime::currentTime();
    QString timeString = currentTime.toString("HHmmss").remove(":");

    // Get Filename
    QFileInfo fileInfo(currentMeshFilePath);
    QDir outputDir = fileInfo.absoluteDir();
    QString outputFilePath = outputDir.filePath(
        QFileInfo(currentMeshFileName).baseName()
        + "_geopotential_"
        + timeString
        + ".txt");

    geopotentialSavePath = outputFilePath;

    // Dispatch Geopotential
    geopotentialDispatcher.setParameter(&mesh, outputFilePath.toStdString());
    geopotentialDispatcher.start();
}

void Widget::onRealRadiusValueChanged(double realRadius) {
    if (isMinMaxAvailable) {
        double scalingFactor = realRadius / min;
        double realDeformity = max * scalingFactor - realRadius;

        ui->minDistanceText->setText(QString::number(min, 'f', 8));
        ui->maxDistanceText->setText(QString::number(max, 'f', 8));
        ui->deformityText->setText(QString::number(max - min, 'f', 8));
        ui->realRadiusDoubleSpinBox->setEnabled(true);
        ui->realDeformityText->setText(QString::number(realDeformity, 'f', 8));
        ui->scalingFactorText->setText(QString::number(scalingFactor, 'f', 8));
    }
}

void Widget::handleMeshLoadingResult(int result) {
    // Reset Progress Bar
    fakeProgressTimer.stop();
    setProgressBar(true, 100);

    if (result == 0) {
        currentMeshFileName = tempMeshFileName;
        currentMeshFilePath = tempMeshFilePath;

        QString select3DModelPushButtonText = tempMeshFileName;
        if (select3DModelPushButtonText.length() > 28) {
            select3DModelPushButtonText = select3DModelPushButtonText.left(25) + "...";
        }
        ui->select3DModelPushButton->setText(select3DModelPushButtonText);
        ui->verticesText->setText(QString::number(mesh.vertexCount()));
        ui->facesText->setText(QString::number(mesh.faceCount()));
        ui->statusText->setText("Ready to generate.");

        isMinMaxAvailable = false;
    }
    else if (result == -2) {
        errorMessageBox("Error", "Mesh contains incomplete faces or N-Gons.");

        ui->select3DModelPushButton->setText(oldSelectButtonText);
        ui->verticesText->setText(oldVerticesText);
        ui->facesText->setText(oldFacesText);
        ui->statusText->setText(oldStatus);
    }
    else {
        errorMessageBox("Error", "Error loading " + tempMeshFilePath);

        ui->select3DModelPushButton->setText(oldSelectButtonText);
        ui->verticesText->setText(oldVerticesText);
        ui->facesText->setText(oldFacesText);
        ui->statusText->setText(oldStatus);
    }

    // Enable UI
    enableUI();
}

void Widget::handleGeopotentialResult(int result, std::string errorString) {
    if (result != 0) {
        if (result == -1) {
            errorMessageBox("Error", "Geopotential parameters not set.");
        }
        else if (result == -2) {
            errorMessageBox("Error", QString::fromStdString(errorString));
        }
        enableUI();
        ui->statusText->setText("Ready to generate.");
        setProgressBar(true, 0);
        return;
    }

    // Enable UI
    enableUI();
    setProgressBar(true, 100);

    // Update Status
    QString statusTextText = "Saved to " + geopotentialSavePath;
    if (statusTextText.length() > 50) {
        statusTextText = statusTextText.left(47) + "...";
    }
    ui->statusText->setText(statusTextText);
}

void Widget::startRenderer() {
    ui->statusText->setText("[2/2] Generating Grayscale Map...");

    fakeProgressTimer.stop();
    setProgressBar(false);

    QString clDevice = ui->clDeviceComboBox->currentText();
    int result = clInterface.init(clDevice.toStdString());
    if (result != 0) {
        if (result == -1) {
            errorMessageBox("Error", "Error initializing OpenCL.");
        }
        else if (result == -2){
            errorMessageBox("Error", "Error compiling OpenCL kernel.");
        }
        else if (result == -3){
            errorMessageBox("Error", "Selected OpenCL device does not support OpenCL 2.0.");
        }
        ui->statusText->setText("Ready to generate.");
        enableUI();
        setProgressBar(true, 0);
        return;
    }

    result = renderer.setBVHBuffer(bvh.getBVH());
    if (result != 0) {
        errorMessageBox("Error", "Error creating OpenCL buffer for BVH.");
        ui->statusText->setText("Ready to generate.");
        enableUI();
        setProgressBar(true, 0);
        return;
    }

    result = renderer.setTriangleBuffer(mesh.getMesh());
    if (result != 0) {
        errorMessageBox("Error", "Error creating OpenCL buffer for triangles.");
        ui->statusText->setText("Ready to generate.");
        enableUI();
        setProgressBar(true, 0);
        return;
    }

    if (ui->colorMapCheckBox->isChecked() && isColorMapLoaded) {
        result = renderer.setColorMapBuffer(colorMapArray, colorMapWidth, colorMapHeight);
        if (result != 0) {
            errorMessageBox("Error", "Error creating OpenCL image buffer for color map.");
            ui->statusText->setText("Ready to generate.");
            enableUI();
            setProgressBar(true, 0);
            return;
        }
    }

    if (ui->normalMapCheckBox->isChecked() && isNormalMapLoaded) {
        result = renderer.setNormalMapBuffer(normalMapArray, normalMapWidth, normalMapHeight);
        if (result != 0) {
            errorMessageBox("Error", "Error creating OpenCL image buffer for normal map.");
            ui->statusText->setText("Ready to generate.");
            enableUI();
            setProgressBar(true, 0);
            return;
        }
    }

    int width = ui->resolutionSpinBox_x->value();
    int height = ui->resolutionSpinBox_y->value();
    bool isRenderColorMap = ui->colorMapCheckBox->isChecked();
    bool isRenderNormalMap = ui->normalMapCheckBox->isChecked();
    renderer.setParameter(width, height, sampleNum, isRenderColorMap, isRenderNormalMap);

    QObject::connect(&renderer,
                     &Renderer::finishedRender,
                     this,
                     &Widget::handleRenderingResult,
                     Qt::UniqueConnection);

    renderer.start();
}

void Widget::handleRenderingResult(int result,
                                   std::shared_ptr<float[]> outputGrayscaleMapArray,
                                   std::shared_ptr<unsigned char[]> outputColorMapArray,
                                   std::shared_ptr<unsigned char[]> outputNormalMapArray,
                                   int width,
                                   int height,
                                   float min,
                                   float max) {

    if (result != 0 || min == 0.0f) {
        if (result == -1) {
            errorMessageBox("Error", "Rendering parameters not set.");
        }
        else if (result == -2) {
            errorMessageBox("Error", "Error executing OpenCL kernel.");
        }
        else if (result == -3) {
            errorMessageBox("Error", "Error reading OpenCL buffer.");
        }
        else if (result == -4) {
            errorMessageBox("Error", "Error creating OpenCL buffer.");
        }
        else if (min == 0.0f) {
            errorMessageBox("Error", "The model is not watertight.");
        }
        ui->statusText->setText("Ready to generate.");
        enableUI();
        setProgressBar(true, 0);
        return;
    }

    ui->statusText->setText("Saving...");

    fakeProgressTimer.stop();
    setProgressBar(true, 100);

    this->min = min;
    this->max = max;

    QFileInfo fileInfo(currentMeshFilePath);
    QDir outputDir = fileInfo.absoluteDir();

    QString grayscaleMapDepthText = ui->grayscaleDepthComboBox->currentText();
    QString grayscaleMapDepthSuffix = "8bit";
    int grayscaleMapDepth = 8;
    if (grayscaleMapDepthText == "16 Bit") {
        grayscaleMapDepth = 16;
        grayscaleMapDepthSuffix = "16bit";
    }

    QString grayscaleMapOutputBaseName = outputDir.filePath(QFileInfo(currentMeshFileName).baseName()
                                                            + "_"
                                                            + "height"
                                                            + "_"
                                                            + grayscaleMapDepthSuffix
                                                            + "_"
                                                            + QString::number(width)
                                                            + "x"
                                                            + QString::number(height));

    QString colorMapOutputBaseName = outputDir.filePath(QFileInfo(currentMeshFileName).baseName()
                                                        + "_"
                                                        + "color"
                                                        + "_"
                                                        + QString::number(width)
                                                        + "x"
                                                        + QString::number(height));

    QString normalMapOutputBaseName = outputDir.filePath(QFileInfo(currentMeshFileName).baseName()
                                                         + "_"
                                                         + "normal"
                                                         + "_"
                                                         + QString::number(width)
                                                         + "x"
                                                         + QString::number(height));

    QTime currentTime = QTime::currentTime();
    QString timeString = currentTime.toString().remove(":");

    QString format = ui->imageFormatComboBox->currentText();

    grayscaleMapSavePath = grayscaleMapOutputBaseName
                           + "_"
                           + timeString
                           + "."
                           + format.toLower();

    colorMapSavePath = colorMapOutputBaseName
                       + "_"
                       + timeString
                       + "."
                       + format.toLower();

    normalMapSavePath = normalMapOutputBaseName
                        + "_"
                        + timeString
                        + "."
                        + format.toLower();

    txtSavePath = grayscaleMapOutputBaseName + "_" + timeString + ".txt";

    QObject::connect(&imageSaver,
                     &ImageSaver::finishedSaving,
                     this,
                     &Widget::handleImageSavingResult,
                     Qt::UniqueConnection);

    bool xFlip = ui->xFlipCheckBox->isChecked();
    bool yFlip = ui->yFlipCheckBox->isChecked();
    double primeMeridian = ui->primeMeridianDoubleSpinBox->value();
    bool isOutputTXT = ui->outputTXTCheckBox->isChecked();

    imageSaver.setParameter(grayscaleMapSavePath.toStdString(),
                            colorMapSavePath.toStdString(),
                            normalMapSavePath.toStdString(),
                            txtSavePath.toStdString(),
                            outputGrayscaleMapArray,
                            outputColorMapArray,
                            outputNormalMapArray,
                            width,
                            height,
                            min,
                            max,
                            format.toStdString(),
                            grayscaleMapDepth,
                            xFlip,
                            yFlip,
                            primeMeridian,
                            isOutputTXT);
    imageSaver.start();
}

void Widget::handleImageSavingResult(int result) {
    if (result != 0) {
        if (result == -1) {
            errorMessageBox("Error", "Image saving parameters not set.");
        }
        else if (result == -2) {
            errorMessageBox("Error", "Error saving grayscale map to " + grayscaleMapSavePath + ".");
        }
        else if (result == -3) {
            errorMessageBox("Error", "Error saving color map to " + colorMapSavePath + ".");
        }
        else if (result == -4) {
            errorMessageBox("Error", "Error saving normal map to " + normalMapSavePath + ".");
        }
        else if (result == -5) {
            errorMessageBox("Error", "Error saving txt to " + txtSavePath + ".");
        }
        ui->statusText->setText("Ready to generate.");
        enableUI();
        setProgressBar(true, 0);
        return;
    }

    isMinMaxAvailable = true;
    enableUI();
    QString statusTextText = "Saved to " + grayscaleMapSavePath;
    if (statusTextText.length() > 50) {
        statusTextText = statusTextText.left(47) + "...";
    }
    ui->statusText->setText(statusTextText);
}

void Widget::updateFakeProgress() {
    int value;
    if (constSpeed) {
        value = std::min(static_cast<int>(t), 98);
        t += step;
    }
    else {
        value = static_cast<int>(2.0f * std::atanf(t) / 3.14159265359f * 100.0f);
        if (value >= 70) {
            t += step * 0.2f;
        }
        else {
            t += step;
        }
    }
    setProgressBar(true, value);
}
