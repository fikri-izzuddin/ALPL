import ij.ImagePlus;
import ij.process.FloodFiller;
import ij.process.ImageProcessor;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        Main main = new Main();

//        for (int i = 1; i < 97; i++) {
//            String url = "input/test_" + String.format("%03d", i) + ".jpg";
//            String[] array = url.split("\\.", -1);
//            String extension = "." + array[array.length - 1];
//
//            String[] array2 = array[0].split("/", 0);
//            String filename = array2[array2.length - 1];
//            filename = "output/" + filename;
//
//            BufferedImage img_rgb = main.readImage(url);
//            BufferedImage img_grayscale = main.convertToGrayscale(img_rgb);
//            BufferedImage img_sharp = main.histogramEqualization(img_grayscale, 256);
//            BufferedImage img_blur = main.gaussianFilter(img_sharp, 7, 1.5);
//            BufferedImage img_edge = main.hysteresis(img_blur, 0.5, main.sobelFilter(img_blur));
//            BufferedImage img_skeleton = main.skeletonize(img_edge);
//            BufferedImage img_morpho = main.selectPlate(img_skeleton);
//
//            int[][] location = main.eightCCL(img_morpho);
//            int x1 = location[0][0];
//            int y1 = location[0][1];
//            int x2 = location[1][0];
//            int y2 = location[1][1];
//
//            int width = x2 - x1;
//            int height = y2 - y1 + 1;
//
//            Graphics2D graphics2D = img_rgb.createGraphics();
//            graphics2D.setColor(Color.RED);
//            graphics2D.setStroke(new BasicStroke(3));
//
//            graphics2D.drawRect(x1, y1, width, height);
//            graphics2D.dispose();
//
//            main.writeImage(img_rgb, filename + extension);
//        }

        String url = "input/test_050.jpg";
        String[] array = url.split("\\.", -1);
        String extension = "." + array[array.length - 1];

        String[] array2 = array[0].split("/", 0);
        String filename = array2[array2.length - 1];
        filename = "output/" + filename;

        System.out.println(filename);
        System.out.println(extension);

        BufferedImage img_rgb = main.readImage(url);
//        main.showImage(img_rgb, "RGB");
        main.writeImage(img_rgb, filename + "_1_rgb" + extension);

        // convert RGB to grayscale
        BufferedImage img_grayscale = main.convertToGrayscale(img_rgb);
//        main.showImage(img_grayscale, "GrayScale");
        main.writeImage(img_grayscale, filename + "_2_grayscale" + extension);

        // sharpening
        BufferedImage img_sharp = main.histogramEqualization(img_grayscale, 256);
//        main.showImage(img_sharp, "Sharped image");
        main.writeImage(img_sharp, filename + "_3_sharp" + extension);

        // blurring
        BufferedImage img_blur = main.gaussianFilter(img_sharp, 7, 1.5);
//        main.showImage(img_blur, "Blurred image");
        main.writeImage(img_blur, filename + "_4_blur" + extension);

        // edge detection
        BufferedImage img_edge = main.hysteresis(img_blur, 0.5, main.sobelFilter(img_blur));
//        main.showImage(img_edge, "Edge");
        main.writeImage(img_edge, filename + "_5_edge" + extension);

        // skeletonize
        BufferedImage img_skeleton = main.skeletonize(img_edge);
//        main.showImage(img_skeleton, "Skeleton");
        main.writeImage(img_skeleton, filename + "_6_skeleton" + extension);

        // morphological operations
        BufferedImage img_morpho = main.selectPlate(img_skeleton);

        // eight neighbourhood connected components labelling (CCL)
        int[][] location = main.eightCCL(img_morpho);

//        main.showImage(img_morpho, "Morpho");
        main.writeImage(img_morpho, filename + "_7_morpho" + extension);

        int x1 = location[0][0];
        int y1 = location[0][1];
        int x2 = location[1][0];
        int y2 = location[1][1];

        int width = x2 - x1;
        int height = y2 - y1 + 1;

        Graphics2D graphics2D = img_rgb.createGraphics();
        graphics2D.setColor(Color.RED);
        graphics2D.setStroke(new BasicStroke(3));

        graphics2D.drawRect(x1, y1, width, height);
        graphics2D.dispose();

//        main.showImage(img_rgb, "Localized");
        main.writeImage(img_rgb, filename + "_8_localized_" + extension);
    }

    // TODO 1. read and show image
    public BufferedImage readImage(String url) {
        BufferedImage img = null;

        try {
            img = ImageIO.read(new File(url));
        } catch (IOException e) {
            System.out.println(e);
        }

        return img;
    }

    public void showImage(BufferedImage img, String label) {
        ImageIcon icon = new ImageIcon(img);

        JFrame frame = new JFrame(label);
        frame.setLayout(new FlowLayout());
        frame.setSize(img.getWidth(), img.getHeight());

        JLabel lbl = new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);

        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public void writeImage(BufferedImage bufferedImage, String url) {
        try {
            ImageIO.write(bufferedImage, "jpg", new File(url));
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    // TODO 2. convert RGB to grayscale
    public BufferedImage convertToGrayscale(BufferedImage img_input) {
        int width = img_input.getWidth();
        int height = img_input.getHeight();

        BufferedImage img_output = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        WritableRaster raster_input = img_input.getRaster();
        WritableRaster raster_output = img_output.getRaster();

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                float gray = 0;
                for (int band = 0; band < raster_input.getNumBands(); band++) {
                    int sample = raster_input.getSample(col, row, band);

                    if (band == 0) { // red channel
                        gray += 0.299 * sample;
                    } else if (band == 1) { // green channel
                        gray += 0.587 * sample;
                    } else if (band == 2) { // blue channel
                        gray += 0.114 * sample;
                        gray = Math.round(gray);
                        raster_output.setSample(col, row, 0, gray);
                    }

                }
            }
        }

        return img_output;
    }

    // TODO 3. sharpening using histogram equalization
    public BufferedImage histogramEqualization(BufferedImage img_input, int intensity) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        BufferedImage img_output = new BufferedImage(width, height, img_input.getType());

        WritableRaster raster_input = img_input.getRaster();
        WritableRaster raster_output = img_output.getRaster();

        int[] histogram = standardHistogram(img_input);
        int[] cumulativeHistogram = cumulativeHistogram(histogram);

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int sample = raster_input.getSample(col, row, 0);
                int equalizedHistogram = cumulativeHistogram[sample] * (intensity - 1) / (width * height);
                raster_output.setSample(col, row, 0, equalizedHistogram);
            }
        }

        return img_output;
    }

    public int[] standardHistogram(BufferedImage img_input) {
        int width = img_input.getWidth();
        int height = img_input.getHeight();

        WritableRaster raster_input = img_input.getRaster();

        int[] histogram = new int[256];

        for (int i = 0; i < histogram.length; i++) {
            histogram[i] = 0;
        }

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int sample = raster_input.getSample(col, row, 0);
                histogram[sample]++;
            }
        }

        return histogram;
    }

    public int[] cumulativeHistogram(int[] histogram) {
        int[] cumulativeHistogram = histogram.clone();

        for (int j = 1; j < cumulativeHistogram.length; j++) {
            cumulativeHistogram[j] = cumulativeHistogram[j - 1] + cumulativeHistogram[j];
        }

        return cumulativeHistogram;
    }

    // TODO 4. smoothing using Gaussian filter
    public BufferedImage gaussianFilter(BufferedImage img_input, int radius, double sigma) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();
        double norm = 0.0;
        double sigma2 = sigma * sigma;
        float[] kernel = new float[2 * radius + 1];

        BufferedImage img_output = new BufferedImage(width, height, img_input.getType());

        WritableRaster raster_input = img_input.getRaster();
        WritableRaster raster_output = img_output.getRaster();

        // create kernel
        for (int x = -radius; x < radius + 1; x++) {
            float exp = (float) Math.exp(-0.5 * (x * x) / sigma2);

            kernel[x + radius] = (float) (1 / (2 * Math.PI * sigma2)) * exp;
            norm += kernel[x + radius];
        }

        // convolve image with kernel horizontally
        for (int row = 0; row < height; row++) {
            for (int col = radius; col < width - radius; col++) {
                double sum = 0.0;

                for (int y = -radius; y < radius; y++) {
                    int sample = raster_input.getSample(col + y, row, 0);
                    sum += (kernel[y + radius] * sample);
                }

                // normalize
                sum /= norm;
                raster_output.setSample(col, row, 0, Math.round(sum));
            }
        }

        // convolve image with kernel vertically
        for (int row = radius; row < height - radius; row++) {
            for (int col = 0; col < width; col++) {
                double sum = 0.0;

                for (int x = -radius; x < radius + 1; x++) {
                    int sample = raster_input.getSample(col, row + x, 0);
                    sum += (kernel[x + radius] * sample);
                }

                // normalize
                sum /= norm;
                raster_output.setSample(col, row, 0, Math.round(sum));
            }
        }

        // retain border value
        for (int row = 0; row < radius; row++) {
            for (int col = 0; col < radius; col++) {
                raster_output.setSample(col, row, 0, raster_input.getSample(col, row, 0));
                raster_output.setSample(width - 1 - col, row, 0,
                        raster_input.getSample(width - 1 - col, row, 0));
                raster_output.setSample(col, height - 1 - row, 0,
                        raster_input.getSample(col, height - 1 - row, 0));
                raster_output.setSample(width - 1 - col, height - 1 - row, 0,
                        raster_input.getSample(width - 1 - col, height - 1 - row, 0));
            }
        }

        return img_output;
    }

    // TODO 5. edge detection with Canny edge detector
    public double[][] sobelFilter(BufferedImage img_input) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        BufferedImage img_output = new BufferedImage(width, height, img_input.getType());

        WritableRaster raster_input = img_input.getRaster();
        WritableRaster raster_output = img_output.getRaster();

        // Sobel filter
        int[][] kernel_horizontal = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
        int[][] kernel_vertical = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };

        int[][] Gy = horizontal(img_input, kernel_horizontal);
        int[][] Gx = vertical(img_input, kernel_vertical);
        double[][] magnitude = magnitude(img_input, Gx, Gy);
        int[][] direction = direction(img_input, Gx, Gy);

        return suppression(img_input, magnitude, direction);
    }

    public int[][] horizontal(BufferedImage img_input, int[][] kernel_horizontal) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        WritableRaster raster_input = img_input.getRaster();

        int[][] Gy = new int[height][width];

        if (height > 2 && width > 2) {
            for (int row = 1; row < height - 1; row++) {
                for (int col = 1; col < width - 1; col++) {
                    int sum = 0;

                    for (int krow = -1; krow < 2; krow++) {
                        for (int kcol = -1; kcol < 2; kcol++) {
                            sum += (kernel_horizontal[krow + 1][kcol + 1] *
                                    raster_input.getSample(col + kcol, row + krow, 0));
                        }
                    }

                    Gy[row][col] = sum;
                }
            }
        }

        return Gy;
    }

    public int[][] vertical(BufferedImage img_input, int[][] kernel_vertical) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        WritableRaster raster_input = img_input.getRaster();

        int[][] Gx = new int[height][width];

        if (height > 2 && width > 2) {
            for (int row = 1; row < height - 1; row++) {
                for (int col = 1; col < width - 1; col++) {
                    int sum = 0;

                    for (int krow = -1; krow < 2; krow++) {
                        for (int kcol = -1; kcol < 2; kcol++) {
                            sum += (kernel_vertical[krow + 1][kcol + 1] *
                                    raster_input.getSample(col + kcol, row + krow, 0));
                        }
                    }

                    Gx[row][col] = sum;
                }
            }
        }

        return Gx;
    }

    public double[][] magnitude(BufferedImage img_input, int[][] Gx, int[][] Gy) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        double sum = 0;
        double var = 0;
        double totalPixel = (height - 1) * (width - 1);
        double[][] magnitude = new double[height][width];

        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                magnitude[row][col] = Math.sqrt((Gx[row][col] * Gx[row][col]) + (Gy[row][col] * Gy[row][col]));
                sum += magnitude[row][col];
            }
        }

        int mean = (int) Math.round(sum / totalPixel);

        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                double diff = magnitude[row][col] - (double) mean;
                var += (diff * diff);
            }
        }

//        int standardDeviation = (int) Math.round(Math.sqrt(var / totalPixel));

        return magnitude;
    }

    public int[][] direction(BufferedImage img_input, int[][] Gx, int[][] Gy) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        double piRad = 180 / Math.PI;
        int[][] direction = new int[height][width];

        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                double theta = Math.atan2(Gy[row][col], Gx[row][col]) * piRad;

                // avoid negative angles
                if (theta < 0) {
                    theta += 360;
                }

                if (theta <= 22.5 || (theta > 157.5 && theta <= 202.5) || theta > 337.5) {
                    direction[row][col] = 0; // left and right
                } else if ((theta > 22.5 && theta <= 67.5) || (theta > 202.5 && theta <= 247.5)) {
                    direction[row][col] = 45; // diagonal: upper right and lower left direction
                } else if ((theta > 67.5 && theta <= 112.5) || (theta > 247.5 && theta <= 292.5)) {
                    direction[row][col] = 90; // top and bottom
                } else {
                    direction[row][col] = 135; // diagonal: upper left and lower right direction
                }
            }
        }

        return direction;
    }

    // TODO 6. non-maximum suppression
    public double[][] suppression(BufferedImage img_input, double[][] magnitude, int[][] direction) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                double suppression_magnitude = magnitude[row][col];

                switch (direction[row][col]) {
                    case 0:
                        if (suppression_magnitude < magnitude[row][col - 1]
                                && suppression_magnitude < magnitude[row][col + 1]) {
                            magnitude[row][col] = 0;
                        }
                        break;
                    case 45:
                        if (suppression_magnitude < magnitude[row - 1][col + 1]
                                && suppression_magnitude < magnitude[row + 1][col - 1]) {
                            magnitude[row][col] = 0;
                        }
                        break;
                    case 90:
                        if (suppression_magnitude < magnitude[row - 1][col]
                                && suppression_magnitude < magnitude[row + 1][col]) {
                            magnitude[row][col] = 0;
                        }
                        break;
                    case 135:
                        if (suppression_magnitude < magnitude[row - 1][col - 1]
                                && suppression_magnitude < magnitude[row + 1][col + 1]) {
                            magnitude[row][col] = 0;
                        }
                        break;
                }
            }
        }

        return magnitude;
    }

    // TODO 7. double thresholding with Otsu's thresholding
    public int otsuThreshold(BufferedImage img_input) {
        int[] histogram = standardHistogram(img_input);
        int totalPixel = img_input.getHeight() * img_input.getHeight();

        float totalPixelValue = 0;

        for (int i = 0; i < histogram.length; i++) {
            totalPixelValue += i * histogram[i];
        }

        int numberBackground = 0;
        int numberForeground = 0;
        float weightBackground = 0;
        float weightForeground = 0;

        float sumBackground = 0;
        float varMax = 0;
        int threshold = 0;

        for (int i = 0; i < 256; i++) {
            numberBackground += histogram[i];
            if (numberBackground == 0) {
                continue;
            }

            numberForeground = totalPixel - numberBackground;
            if (numberForeground == 0) {
                break;
            }

            sumBackground += (float) (i * histogram[i]);

            weightBackground = (float) numberBackground / (float) totalPixel;
            weightForeground = (float) numberForeground / (float) totalPixel;

            float meanB = sumBackground / (float) numberBackground;
            float meanF = (totalPixelValue - sumBackground) / (float) numberForeground;

            float varBetween = weightBackground * weightForeground * (meanB - meanF) * (meanB - meanF);

            if (varBetween > varMax) {
                varMax = varBetween;
                threshold = i;
            }
        }

        return threshold;
    }

    // TODO 8. edge tracking by hysteresis
    public BufferedImage hysteresis(BufferedImage img_input, double tRatio, double[][] magnitude) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        BufferedImage img_output = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        WritableRaster raster_output = img_output.getRaster();

        double tHigh = otsuThreshold(img_input);
        double tLow = tHigh * tRatio;

        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                double hysteresisMagnitude = magnitude[row][col];
                int sample = 0;

                if (hysteresisMagnitude >= tHigh) {
                    sample = 1;
                } else if (hysteresisMagnitude < tLow) {
                    sample = 0;
                } else {
                    boolean connected = false;

                    for (int krow = -1; krow < 2; krow++) {
                        for (int kcol = -1; kcol < 2; kcol++) {
                            if (magnitude[row + krow][col + kcol] >= tHigh) {
                                connected = true;
                            }
                        }
                    }

                    sample = (connected) ? 1 : 0;
                }

                raster_output.setSample(col, row, 0, sample);
            }
        }

        return img_output;
    }

    // TODO 9. skeletonize
    public BufferedImage skeletonize(BufferedImage img_input) {
        int height = img_input.getHeight();
        int width = img_input.getWidth();

        BufferedImage img_output = new BufferedImage(width, height, img_input.getType());

        WritableRaster raster_input = img_input.getRaster();
        WritableRaster raster_output = img_output.getRaster();

        int[][] firstArray = new int[height + 2][width + 2];
        int[][] secondArray = new int[height + 2][width + 2];

        boolean changeFlag = true;

        for (int row = 0; row < height + 2; row++) {
            for (int col = 0; col < width + 2; col++) {
                firstArray[row][col] = 0;
                secondArray[row][col] = 0;
            }
        }

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int sample = raster_input.getSample(col, row, 0);
                firstArray[row + 1][col + 1] = sample;
                secondArray[row + 1][col + 1] = sample;
            }
        }

        while (changeFlag) {
            changeFlag = false;

            // north thinning
            for (int row = 1; row < height + 1; row++) {
                for (int col = 1; col < width + 1; col++) {
                    if ((firstArray[row][col] > 0) && (firstArray[row - 1][col] == 0)) {
                        if (thinning(row, col, firstArray)) {
                            secondArray[row][col] = 0;
                            changeFlag = true;
                        } else {
                            secondArray[row][col] = 1;
                            changeFlag = false;
                        }
                    }
                }
            }

            firstArray = secondArray.clone();

            // south thinning
            for (int row = 1; row < height + 1; row++) {
                for (int col = 1; col < width + 1; col++) {
                    if ((firstArray[row][col] > 0) && (firstArray[row + 1][col] == 0)) {
                        if (thinning(row, col, firstArray)) {
                            secondArray[row][col] = 0;
                            changeFlag = true;
                        } else {
                            secondArray[row][col] = 1;
                            changeFlag = false;
                        }
                    }
                }
            }

            firstArray = secondArray.clone();

            // west thinning
            for (int row = 1; row < height + 1; row++) {
                for (int col = 1; col < width + 1; col++) {
                    if ((firstArray[row][col] > 0) && (firstArray[row][col - 1] == 0)) {
                        if (thinning(row, col, firstArray)) {
                            secondArray[row][col] = 0;
                            changeFlag = true;
                        } else {
                            secondArray[row][col] = 1;
                            changeFlag = false;
                        }
                    }
                }
            }

            firstArray = secondArray.clone();

            // east thinning
            for (int row = 1; row < height + 1; row++) {
                for (int col = 1; col < width + 1; col++) {
                    if ((firstArray[row][col] > 0) && (firstArray[row][col + 1] == 0)) {
                        if (thinning(row, col, firstArray)) {
                            secondArray[row][col] = 0;
                            changeFlag = true;
                        } else {
                            secondArray[row][col] = 1;
                            changeFlag = false;
                        }
                    }
                }
            }

            firstArray = secondArray.clone();
        }

        for (int row = 1; row < height + 1; row++) {
            for (int col = 1; col < width + 1; col++) {
                int sample = firstArray[row][col];
                if (sample == 0) {
                    sample = 1;
                } else {
                    sample = 0;
                }

                raster_output.setSample(col - 1, row - 1, 0, sample);
            }
        }

        return img_output;
    }

    public boolean thinning(int row, int col, int[][] fistArray) {
        int nonZero = -1;
        boolean valid = false;

        for (int krow = -1; krow <= 1; krow++) {
            for (int kcol = -1; kcol <= 1; kcol++) {
                if (fistArray[row + krow][col + kcol] != 0) {
                    nonZero++;
                }
            }
        }

        int p1 = fistArray[row - 1][col - 1];
        int p2 = fistArray[row - 1][col];
        int p3 = fistArray[row - 1][col + 1];
        int p4 = fistArray[row][col - 1];
        int p5 = fistArray[row][col + 1];
        int p6 = fistArray[row + 1][col - 1];
        int p7 = fistArray[row + 1][col];
        int p8 = fistArray[row + 1][col + 1];

        if ((p2 == 0 && p7 == 0) ||
                (p4 == 0 && p5 == 0) ||
                (p1 == 1 && p2 == 0 && p4 == 0) ||
                (p5 == 0 && p7 == 0 && p8 == 1) ||
                (p2 == 0 && p3 == 1 && p5 == 0) ||
                (p4 == 0 && p6 == 1 && p7 == 0)) {
            valid = true;
        }

        if ((p1 == 0 && p8 == 0 && ((p5 != 1 && p7 != 1) || (p2 != 1 && p4 != 1))) ||
                (p3 == 0 && p6 == 0 && ((p2 != 1 && p4 != 1) || (p5 != 1 && p7 != 1)))) {
            valid = true;
        }

        if (nonZero >= 4 && valid == false) {
            return true;
        } else {
            return false;
        }
    }

    // TODO 10. morphological operation
    public void fillHole(ImageProcessor imageProcessor) {
        int height = imageProcessor.getHeight();
        int width = imageProcessor.getWidth();

        int foreground = 0;
        int background = 1;
        imageProcessor.setSnapshotCopyMode(true);

        FloodFiller floodFiller = new FloodFiller(imageProcessor);
        imageProcessor.setColor(127);

        for (int y = 0; y < height; y++) {
            if (imageProcessor.getPixel(0, y) == background) {
                floodFiller.fill(0, y);
            }

            if (imageProcessor.getPixel(width - 1, y) == background) {
                floodFiller.fill(width - 1, y);
            }
        }

        for (int x = 0; x < width; x++) {
            if (imageProcessor.getPixel(x, 0) == background) {
                floodFiller.fill(x, 0);
            }

            if (imageProcessor.getPixel(x, height - 1) == background) {
                floodFiller.fill(x, height - 1);
            }
        }

        byte[] pixels = (byte[]) imageProcessor.getPixels();
        int pixelCount = width * height;

        for (int i = 0; i < pixelCount; i++) {
            if (pixels[i] == 127) {
                pixels[i] = (byte) background;
            } else {
                pixels[i] = (byte) foreground;
            }
        }

        imageProcessor.setSnapshotCopyMode(false);
        imageProcessor.setBinaryThreshold();
    }

    public void erode(ImageProcessor imageProcessor, int loop) {
        for (int i = 0; i < loop; i++) {
            imageProcessor.erode();
        }
    }

    public void dilate(ImageProcessor imageProcessor, int loop) {
        for (int i = 0; i < loop; i++) {
            imageProcessor.dilate();
        }
    }

    public void opening(ImageProcessor imageProcessor, int loop) {
        for (int i = 0; i < loop; i++) {
            imageProcessor.erode();
            imageProcessor.dilate();
        }
    }

    public void closing(ImageProcessor imageProcessor, int loop) {
        for (int i = 0; i < loop; i++) {
            imageProcessor.dilate();
            imageProcessor.erode();
        }
    }

    public BufferedImage invertColor(BufferedImage img) {
        WritableRaster raster = img.getRaster();

        for (int row = 0; row < raster.getHeight(); row++) {
            for (int col = 0; col < raster.getWidth(); col++) {
                int sample = raster.getSample(col, row, 0);

                if (sample == 0) {
                    sample = 1;
                } else {
                    sample = 0;
                }

                raster.setSample(col, row, 0, sample);
            }
        }

        return img;
    }

    // TODO 11. eight neighbourhood connected components labelling (CCL)
    public int[][] eightCCL(BufferedImage img) {
        int height = img.getHeight();
        int width = img.getWidth();

        WritableRaster raster = img.getRaster();

        int[][] imgArray = new int[height + 2][width + 2];
        int newLabel = 0;
        int[] eqArray = new int[(height * width) / 2];

        for (int i = 0; i < eqArray.length; i++) {
            eqArray[i] = 0;
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                imgArray[i][j] = 0;
            }
        }

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                imgArray[row + 1][col + 1] = raster.getSample(col, row, 0);
            }
        }

        // 1st pass
        for (int row = 1; row < height + 1; row++) {
            for (int col = 1; col < width + 1; col++) {
                int pixel = imgArray[row][col];

                if (pixel > 0) {
                    int NW = imgArray[row - 1][col - 1];
                    int N = imgArray[row - 1][col];
                    int NE = imgArray[row - 1][col + 1];
                    int W = imgArray[row][col - 1];

                    if (NW == 0 && N == 0 && NE == 0 && W == 0) {
                        newLabel += 1;
                        imgArray[row][col] = newLabel;
                    }

                    else if ((NW != 0 || N != 0 || NE != 0 || W != 0) &&
                            ((NW != 0 && (NW == N || NW == NE || NW == W)) ||
                                    (N != 0 && (N == NE || N == W)) ||
                                    (NE != 0 && (NE == W)))) {
                        int tmp = -1;
                        if (NW != 0)
                            tmp = NW;
                        else if (N != 0)
                            tmp = N;
                        else if (NE != 0)
                            tmp = NE;
                        else if (W != 0)
                            tmp = W;
                        imgArray[row][col] = tmp;
                    }

                    else if (NW != 0 || N != 0 || NE != 0 || W != 0) {
                        int min = newLabel;
                        int max = NW;

                        if (NW != 0 && NW < min)
                            min = NW;
                        if (N > max)
                            max = N;
                        if (N != 0 && N < min)
                            min = N;
                        if (NE > max)
                            max = NE;
                        if (NE != 0 && NE < min)
                            min = NE;
                        if (W > max)
                            max = W;
                        if (W != 0 && W < min)
                            min = W;

                        imgArray[row][col] = min;
                        eqArray[max] = min; // Linked or Union
                    }
                }
            }
        }

        // 2nd pass
        for (int row = height; row > 1; row--) {
            for (int col = width; col > 1; col--) {
                int pixel = imgArray[row][col];

                if (pixel > 0) {

                    int E = imgArray[row][col + 1]; // East
                    int SW = imgArray[row + 1][col - 1]; // South-West
                    int S = imgArray[row + 1][col]; // South
                    int SE = imgArray[row + 1][col + 1]; // South-East

                    if ((E != pixel && E != 0) || (SW != pixel && SW != 0) ||
                            (S != pixel && S != 0) || (SE != pixel && SE != 0)) {
                        int min = pixel;
                        int max = pixel;
                        if (E != 0 && E < min)
                            min = E;
                        if (E > max)
                            max = E;
                        if (SW != 0 && SW < min)
                            min = SW;
                        if (SW > max)
                            max = SW;
                        if (S != 0 && S < min)
                            min = S;
                        if (S > max)
                            max = S;
                        if (SE != 0 && SE < min)
                            min = SE;

                        imgArray[row][col] = min;
                        eqArray[max] = min;
                    }

                }
            }
        }

        // arrange eq
        int count = 0;
        for (int i = 1; i < newLabel + 1; i++) {
            if (eqArray[i] == i) {
                count++;
                eqArray[i] = count;
            } else {
                eqArray[i] = eqArray[eqArray[i]];
            }
        }

        // 3rd pass
        int[] pixelCount = new int[count + 1];
        BoundingBox[] boxes = new BoundingBox[count + 1];

        for (int i = 0; i < count + 1; i++) {
            pixelCount[i] = 0;
            boxes[i] = new BoundingBox((height * width) / 4);
        }

        for (int row = 1; row < height + 1; row++) {
            for (int col = 1; col < width + 1; col++) {
                int pixel = imgArray[row][col];

                if (pixel > 0) {
                    if (pixel != eqArray[pixel]) {
                        imgArray[row][col] = eqArray[pixel];
                    }

                    if (boxes[imgArray[row][col]].minrow > row) {
                        boxes[imgArray[row][col]].minrow = row - 1;
                    }

                    if (boxes[imgArray[row][col]].mincol > col) {
                        boxes[imgArray[row][col]].mincol = col - 1;
                    }

                    if (boxes[imgArray[row][col]].maxrow < row) {
                        boxes[imgArray[row][col]].maxrow = row - 1;
                    }

                    if (boxes[imgArray[row][col]].maxcol < col) {
                        boxes[imgArray[row][col]].maxcol = col - 1;
                    }
                }

                pixelCount[imgArray[row][col]]++;
            }
        }

        // localize
        int[][] licensePlate = new int[2][2];

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                licensePlate[i][j] = 0;
            }
        }

        int tmp_index = -1;
        int pixelObj = 0;

        for (int i = 1; i < boxes.length; i ++) {
            int lpHeight = boxes[i].maxrow - boxes[i].minrow + 1;
            int lpWidth = boxes[i].maxcol - boxes[i].mincol + 1;

            if (lpHeight > 81) {
                continue;
            }

            double ratio = (double) lpWidth / (double) lpHeight;

            if (ratio < 2.9 &&
                    ratio > 2 &&
                    pixelCount[i] >= 20 &&
                    pixelCount[i] <= 15000 &&
                    lpHeight >= 20 &&
                    lpWidth >= 40)  {
                if (pixelObj < pixelCount[i]) {
                    pixelObj = pixelCount[i];
                    tmp_index = i;
                }
            }

            if (ratio <= 6 && ratio >= 2.9) {
                int size = lpWidth * lpHeight;
                if (( (double) pixelCount[i] / (double) size) >= 0.6) {
                    if (pixelCount[i] < 20 ||
                            pixelCount[i] > 15000 ||
                            lpHeight < 20 ||
                            lpWidth < 40) {
                        continue;
                    }

                    licensePlate[0][0] = boxes[i].mincol;
                    licensePlate[0][1] = boxes[i].minrow;
                    licensePlate[1][0] = boxes[i].maxcol;
                    licensePlate[1][1] = boxes[i].maxrow;
                }
            }

        }

        if (licensePlate[0][0] == 0 &&
                licensePlate[0][1] == 0 &&
                licensePlate[1][0] == 0 &&
                licensePlate[1][1] == 0 &&
                tmp_index > -1 && pixelObj > 0) {
            licensePlate[0][0] = boxes[tmp_index].mincol;
            licensePlate[0][1] = boxes[tmp_index].minrow;
            licensePlate[1][0] = boxes[tmp_index].maxcol;
            licensePlate[1][1] = boxes[tmp_index].maxrow;
        }

        return licensePlate;

    }

    public class BoundingBox {
        int minrow;
        int mincol;
        int maxrow;
        int maxcol;

        public BoundingBox(int count) {
            minrow = count;
            mincol = count;
            maxrow = 0;
            maxcol = 0;
        }
    }

    // TODO 12. select plate
    public BufferedImage selectPlate(BufferedImage img_input) {
        BufferedImage img_output = morpho1pass(img_input);
        int[][] location = eightCCL(img_output);
        double ratio;

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho2pass(img_input);
        location = eightCCL(img_output);

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho3pass(img_input);
        location = eightCCL(img_output);

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho4pass(img_input);
        location = eightCCL(img_output);

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho5pass(img_input);
        location = eightCCL(img_output);

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho6pass(img_input);
        location = eightCCL(img_output);

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho7pass(img_input);
        location = eightCCL(img_output);

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho8pass(img_input);
        location = eightCCL(img_output);

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho9pass(img_input);
        location = eightCCL(img_output);

        if (location[0][0] != 0 || location[0][1] != 0 || location[1][0] != 0 || location[1][1] != 0) {
            return img_output;
        }

        img_output = morpho10pass(img_input);
        return img_output;
    }

    private BufferedImage morpho1pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        fillHole(imageProcessor);
        opening(imageProcessor, 1);
        closing(imageProcessor, 1);
        fillHole(imageProcessor);
        erode(imageProcessor, 10);
        dilate(imageProcessor, 10);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho2pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        closing(imageProcessor, 1);
        dilate(imageProcessor, 3);
        erode(imageProcessor, 10);
        dilate(imageProcessor, 7);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho3pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        closing(imageProcessor, 1);
        dilate(imageProcessor, 3);
        closing(imageProcessor, 1);
        erode(imageProcessor, 10);
        dilate(imageProcessor, 7);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho4pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        fillHole(imageProcessor);
        dilate(imageProcessor, 1);
        erode(imageProcessor, 10);
        dilate(imageProcessor, 7);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho5pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        fillHole(imageProcessor);
        closing(imageProcessor, 1);
        fillHole(imageProcessor);
        erode(imageProcessor, 10);
        dilate(imageProcessor, 10);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho6pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        fillHole(imageProcessor);
        dilate(imageProcessor, 1);
        closing(imageProcessor, 1);
        fillHole(imageProcessor);
        opening(imageProcessor, 1);
        erode(imageProcessor, 10);
        dilate(imageProcessor, 9);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho7pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        dilate(imageProcessor, 1);
        closing(imageProcessor, 1);
        erode(imageProcessor, 1);
        opening(imageProcessor, 1);
        fillHole(imageProcessor);
        erode(imageProcessor, 10);
        dilate(imageProcessor, 10);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho8pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        fillHole(imageProcessor);
        dilate(imageProcessor, 3);
        closing(imageProcessor, 1);
        erode(imageProcessor, 3);
        opening(imageProcessor, 1);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho9pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        fillHole(imageProcessor);
        opening(imageProcessor, 1);
        dilate(imageProcessor, 3);
        closing(imageProcessor, 1);
        fillHole(imageProcessor);
        erode(imageProcessor, 10);
        dilate(imageProcessor, 8);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    private BufferedImage morpho10pass(BufferedImage img_input) {
        ImagePlus imagePlus = new ImagePlus("Image", img_input);
        ImageProcessor imageProcessor = imagePlus.getProcessor();

        fillHole(imageProcessor);
        opening(imageProcessor, 1);
        erode(imageProcessor, 5);
        dilate(imageProcessor, 5);

        BufferedImage img_output = imagePlus.getBufferedImage();
        return invertColor(img_output);
    }

    BufferedImage deepCopy(BufferedImage bi) {
        ColorModel cm = bi.getColorModel();
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = bi.copyData(null);
        return new BufferedImage(cm, raster, isAlphaPremultiplied, null);
    }
}