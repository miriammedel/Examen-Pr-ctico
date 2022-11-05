
//Medel Ortiz Miriam
// 5BV1
//Primer Examen Parcial
//Visión Artificial

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
//Se define pi y e para los calculos
#define pi 3.1416
#define e 2.72

using namespace cv;
using namespace std;

//Función para generar el kernel
//Creación del vector vacio tipo flotante para el kernel
vector<vector<float>> Kernel(int k, int sigma) {
	//Centro del kernel
	int centro = (k - 1) / 2;
	vector<vector<float>> v(k, vector<float>(k, 0));
	for (int i = -centro; i <= centro; i++){
		for (int j = -centro; j <= centro; j++){
			float resultado = (1 / (2 * pi * sigma * sigma)) * pow(e, -((i * i + j * j) / (2 * sigma * sigma)));
			v[i + centro][j + centro] = resultado;
		}
	}
	return v;
}

//Aplicación del filtro en filas y columnas de la matriz de la imagen con el kernel solicitado
float filtro(Mat original, vector<vector<float>> kernel, int k, int x, int y) {
	//Declaración de filas y columnas
	int rows = original.rows;
	int cols = original.cols;
	//Centro
	int centrof = (k - 1) / 2;
	float sumFilter = 0;
	float sumKernel = 0;
	for (int i = -centrof; i <= centrof; i++){

		//Recorre filas y columnas de la imagen a partir del centro
		for (int j = -centrof; j <= centrof; j++){
			float kTmp = kernel[i + centrof][j + centrof];
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			if (!(tmpX < 0 || tmpX >= cols || tmpY < 0 || tmpY >= rows)) {
				tmp = original.at<uchar>(Point(tmpX, tmpY));
			}

			sumFilter += (kTmp * tmp);
			sumKernel += kTmp;
		}
	}
	return sumFilter / sumKernel;
}

//Se aplica el filtro a la imagen
Mat filtro_imagen(Mat original, vector<vector<float>> kernel, int k) {
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	for (int i = 0; i < original.rows; i++) {

		for (int j = 0; j < original.cols; j++) {

			filteredImg.at<uchar>(Point(i, j)) = uchar(filtro(original, kernel, k, i, j));
		}
	}
	return filteredImg;
}

//Función donde se lee el tamaño de las inagenes
void dimensiones(Mat imagen);
int main() {
	char NombreImagen[] = "Lena.png";

	// Matriz de la imagen
	Mat imagen, Image_ecualiz;

	int sigma;
	int k;

	//Se pide el tamaño del kernel y se guarda en la variable k
	cout << "Ingresa el tamano del kernel, debe ser un numero impar" << endl;
	cin >> k;

	//Si el valor ingresado es par, manda una advertencia y no continúa el proceso
	if (k % 2 == 0) {
		cout << "Valor de kernel invalido" << endl;
		exit(0);
	}

	//Se pide un valor de sigma y se guarda en la variable sigma
	cout << "Ingresa sigma" << endl;
	cin >> sigma;

	// Matriz de la imagen 2
	Mat imagenGrises;
	int i, j;
	double azul, verde, rojo;

	// Leer imagen
	imagen = imread(NombreImagen);
	if (!imagen.data) {
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	// No. de filas de imagen
	int filasOriginal = imagen.rows;

	// No. de columnas de imagen
	int columnasOriginal = imagen.cols;

	// CV_8UC1 Dato de tipo uchar(contener solo un carácter único) para la matriz de un solo canal
	Mat imagenGrisesPromedio(filasOriginal, columnasOriginal, CV_8UC1);
	Mat imagenGrisesNTSC(filasOriginal, columnasOriginal, CV_8UC1);

	//Función para cambiar la imagen a escala de grises
	for (i = 0; i < filasOriginal; i++) {
		for (j = 0; j < columnasOriginal; j++) {
			//Vamos a obtener los bits azul, verde y rojo de la correspondiente imagen
			//point accede al valor del pixel en el punto(j,i)
			//Promedio
			azul = imagen.at<Vec3b>(Point(j, i)).val[0];  // B
			verde = imagen.at<Vec3b>(Point(j, i)).val[1]; // G
			rojo = imagen.at<Vec3b>(Point(j, i)).val[2];  // R

			//NTSC
			imagenGrisesPromedio.at<uchar>(Point(j, i)) = uchar((azul + verde + rojo) / 3);
			imagenGrisesNTSC.at<uchar>(Point(j, i)) = uchar(0.299 * azul + 0.587 * verde + 0.11 * rojo);
		}
	}

	vector<vector<float>> kernel = Kernel(k, sigma);
	Mat filtrada = filtro_imagen(imagenGrisesNTSC, kernel, k);

	//Ecualización

	//Variables para el histograma
	int histograma = 256;
	// el rango del nivel del gris 0-255
	float range[] = { 0, 256 };
	const float* histogramaR = {range};

	//Visualización del histograma
	int ancho = 512; int alto = 400;
	int bin_an = cvRound((double)ancho / histograma);
	// CV_8UC3 indica que el rango de valores estará entre 0 - 255
	Mat histImage(alto, ancho, CV_8UC3, Scalar(0, 0, 0));
	Mat ecualizacion(alto, ancho, CV_8UC3, Scalar(0, 0, 0));

	//calculo del histograma
	Mat original, normalizado, ecualizado, ecua_nom;
	calcHist(&imagenGrisesNTSC, 1, 0, Mat(), original, 1, &histograma, &histogramaR, true, false);

	// Normalizar el resultado a [ 0, histImage.rows ]
	normalize(original, normalizado, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//Ecualizacion del histograma a partir de una imagen en escala de grises	
	equalizeHist(imagenGrisesNTSC, Image_ecualiz);
	calcHist(&Image_ecualiz, 1, 0, Mat(), ecualizado, 1, &histograma, &histogramaR, true, false);


	//Normalizar el histograma ecualizado
	normalize(ecualizado, ecua_nom, 0, histImage.rows, NORM_MINMAX, -1, Mat());


	// Creacion de una ventana
	namedWindow("Imagen original", WINDOW_AUTOSIZE);
	//Mostrar la imagen original
	imshow("Imagen original", imagen);
	//Mostrar el tamaño de la imagen
	dimensiones(imagen);

	//Creación de una ventana
	namedWindow("Imagen escala de grises (Promedio)", WINDOW_AUTOSIZE);
	//Mostrar la imagen a escala de grises Promedio
	imshow("Imagen escala de grises (Promedio)", imagenGrisesPromedio);
	//Mostrar el tamaño de la imagen
	dimensiones(imagenGrisesPromedio);

	//Creación de una ventana
	namedWindow("Imagen escala de grises (NTSC)", WINDOW_AUTOSIZE);
	//Mostrar la imagen a escala de grises NTSC
	imshow("Imagen escala de grises (NTSC)", imagenGrisesNTSC);
	//Mostrar el tamaño de la imagen
	dimensiones(imagenGrisesNTSC);

	//Creación de una ventana
	namedWindow("Imagen Gauss", WINDOW_AUTOSIZE);
	//Mostrar la imagen con filtro de Gauss
	imshow("Imagen Gauss", filtrada);
	//Mostrar el tamaño de la imagen
	dimensiones(filtrada);

	//Creación de una ventana
	namedWindow("Imagen ecualizada", WINDOW_AUTOSIZE);
	//Mostrar la imagen ecualizada
	imshow("Imagen ecualizada", Image_ecualiz);
	//Mostrar el tamaño de la imagen
	dimensiones(Image_ecualiz);

	waitKey(0); // Funcion para esperar

}

//Función para mostrar los tamaños de imagen
void dimensiones(Mat imagen) {
	int fila = 0, columna = 0;
	fila = imagen.rows;
	columna = imagen.cols;
	printf("\nPixeles largo %d", fila);
	printf("\nPixeles ancho %d", columna);
}
