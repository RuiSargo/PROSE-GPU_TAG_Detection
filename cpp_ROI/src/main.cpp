#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <iostream>
#include <locale>

int main() {
    // Configurar a codificação para suportar caracteres com acentos no terminal
    std::setlocale(LC_ALL, "pt_PT.UTF-8");
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Inicializar medidor de tempo
    cv::TickMeter tm;

    // Carregar imagem
    std::string imagePath = "C:\\Users\\RUI\\ISEP\\MEEC\\2_ano\\PROSE\\TrabFinal\\Code\\cpp_ROI\\aruco_images\\original_world_aruco_test2.jpg"; //caminho da imagem
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Erro ao carregar a imagem!" << std::endl;
        return -1;
    }

    // Converter para escala de cinza
    tm.start();
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    tm.stop();
    std::cout << "Tempo para conversão para escala de cinza: " << tm.getTimeMilli() << " ms" << std::endl;
    tm.reset();
    // Exibir imagem em escala de cinza
    cv::namedWindow("Imagem GRAYSCALE", cv::WINDOW_NORMAL);  // Permitir redimensionamento da janela
    cv::imshow("Imagem GRAYSCALE", gray);
    cv::waitKey(1);  // Atualiza a janela sem bloquear o código

    // Aplicar filtro para reduzir ruído
    tm.start();
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    tm.stop();
    std::cout << "Tempo para aplicação de filtro Gaussiano: " << tm.getTimeMilli() << " ms" << std::endl;
    tm.reset();
    // Exibir imagem borrada
    cv::namedWindow("Imagem Blured", cv::WINDOW_NORMAL);  // Permitir redimensionamento da janela
    cv::imshow("Imagem Blured", blurred);
    cv::waitKey(1);  // Atualiza a janela sem bloquear o código

    // Deteção de bordas
    tm.start();
    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);
    tm.stop();
    std::cout << "Tempo para deteção de bordas: " << tm.getTimeMilli() << " ms" << std::endl;
    tm.reset();
    // Exibir imagem com bordas
    cv::namedWindow("Imagem com Bordas", cv::WINDOW_NORMAL);  // Permitir redimensionamento da janela
    cv::imshow("Imagem com Bordas", edges);
    cv::waitKey(1);  // Atualiza a janela sem bloquear o código

    // Inicializar o dicionário ArUco e o detetor
    // Usar o objeto diretamente, não o ponteiro
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    cv::aruco::DetectorParameters parameters;
    cv::aruco::ArucoDetector detector(dictionary, parameters);

    // Detetar marcadores ArUco
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;

    tm.start();
    detector.detectMarkers(image, corners, ids);
    tm.stop();
    std::cout << "Tempo para deteção de marcadores ArUco: " << tm.getTimeMilli() << " ms" << std::endl;

    // Exibir resultados
    if (!ids.empty()) {
        std::cout << "Marcadores detetados: " << ids.size() << std::endl;
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << "ID: " << ids[i] << ", Cantos: ";
            for (const auto& point : corners[i]) {
                std::cout << "(" << point.x << ", " << point.y << ") ";
            }
            std::cout << std::endl;
        }

        // Desenhar os marcadores na imagem
        cv::aruco::drawDetectedMarkers(image, corners, ids);

        // Exibir a imagem final com marcadores
        cv::namedWindow("Marcadores ArUco Detetados", cv::WINDOW_NORMAL);  // Permitir redimensionamento da janela
        cv::imshow("Marcadores ArUco Detetados", image);
        cv::waitKey(1);  // Atualiza a janela sem bloquear o código
    } else {
        std::cout << "Nenhum marcador ArUco foi detetado." << std::endl;
    }

    // Permitir que todas as janelas sejam fechadas após a execução do código
    cv::waitKey(0);  // Esperar até o usuário pressionar uma tecla para fechar todas as janelas

    return 0;
}
