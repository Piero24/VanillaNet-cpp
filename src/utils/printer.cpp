#include "printer.hpp"


void clearScreen()
{
    #ifdef _WIN32
        system("cls"); // For Windows
    #else
        system("clear"); // For Unix/Linux
    #endif
}


int getTerminalWidth()
{
    #ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
    #else
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        return w.ws_col;
    #endif
}

void printCentered(const std::string &text, char c)
{
    int termWidth = getTerminalWidth();
    int textWidth = text.length();
    int leadingSpaces = (termWidth - textWidth - 2) / 2; // Subtracting 2 for the characters on both sides

    // Print leading characters
    for (int i = 0; i < leadingSpaces; i++)
    {
        std::cout << c;
    }

    // Print text
    std::cout << text;

    if (c != ' ')
    {
        // Print trailing characters
        for (int i = 0; i < leadingSpaces; i++)
        {
            std::cout << c;
        }

        // If the width is odd, print an additional trailing character
        if ((termWidth - textWidth - 2) % 2 != 0)
        {
            std::cout << c;
        }
    }

    std::cout << std::endl;
}


void printHorizontalLine(char c)
{
    int termWidth = getTerminalWidth();
    for (int i = 0; i < termWidth; i++)
    {
        putchar(c);
    }
    putchar('\n');
}


void infoPrinter(Arguments& inputParams, Network& net)
{   
    clearScreen();
    printCentered("   WELCOME TO THE VANILLANET-CPP  ", '-');
    printf("\n");
    
    printCentered("A neural network build from scratch in C++", ' ');
    printf("\n\n");

    printCentered("Andrea Pietrobon - MIT License", ' ');
    printHorizontalLine('-');

    printf("\n");
    printCentered("      SELECTED PARAMETERS      ", ' ');
	printCentered("*******************************", ' ');
	printf("\n");


    std::cout << "- Mode:                        ";
    if (inputParams.Train)
        std::cout << "Training";

    if (inputParams.Test)
    {
        if (inputParams.Train)
            std::cout << " & ";
        std::cout << "Testing";
    }
    std::cout << "\n" << std::endl;

    if (inputParams.Train)
    {
        std::cout << "- Training dataset size:       " << inputParams.TrainDatasetImages.size() << std::endl;
        std::cout << "- Training dataset:            " << inputParams.TrainDatasetPath << std::endl;
    }
    
    if (inputParams.Test)
    {
        std::cout << "- Testing dataset size:        " << inputParams.TestDatasetImages.size() << std::endl;
        std::cout << "- Testing dataset:             " << inputParams.TestDatasetPath << std::endl;
    }

    std::cout << "\n- Import Weights and biases:   " << (inputParams.hasWeightsBiases ? "True" : "False") << std::endl;
    if (inputParams.hasWeightsBiases)
    {
        std::cout << "- Weights and biases path:     " << inputParams.WeightsBiasesPath << std::endl;
    }

    std::cout << "\n- Network Type:                Fully Connected (FC)" << std::endl;
    std::cout << "- Number of layers:            " << net.standardLayerCount << std::endl;
    std::cout << "- Type of layers:";

    int j = 1;
    for (int i = 0; i < net.Layers.size(); i++)
    {
        // Identify the type of layer using the polymorphic method getType
        if (net.Layers[i]->getType() == LayerType::StandardLayer)
        {
            std::cout << std::endl << "                               " << j << ") F.C. input: " << net.Layers[i]->inputSize << " neurons: " << net.Layers[i]->outputSize;
            j++;
        }
        else if (net.Layers[i]->getType() == LayerType::ActivationLayer)
        {
            // Use dynamic_cast to access the ActivationLayer
            ActivationLayer* activationLayer = dynamic_cast<ActivationLayer*>(net.Layers[i].get());
            if (activationLayer != nullptr)
            {
               std::cout << " A.F.: " << ActivationTypeToString(activationLayer->activationFunction);
            }
        }
    }
    std::cout << "\n" << std::endl;
    std::cout << "- Loss:                        Min Squared Error Loss" << std::endl;
    
    if (inputParams.Train)
    {
        std::cout << "- Optimizer:                   (SGD) Stochastic Gradient Descent" << std::endl;
        std::cout << "\n- Epochs:                      " << inputParams.epochs << std::endl;
        std::cout << "- Batch Size:                  " << inputParams.batchSize << std::endl;
        std::cout << "- Learning Rate:               " << inputParams.learningRate << std::endl;
    }

    printf("\n");
	printHorizontalLine('*');
    printf("\n");
}

void finalResultPrinter(double accuracy, double loss, int corrects, int total, std::string title)
{
    int termWidth = getTerminalWidth();

    std::ostringstream ossAcc, ossLoss;
    ossAcc << std::fixed << std::setprecision(2) << accuracy;
    ossLoss << std::fixed << std::setprecision(5) << loss;

    std::string correctClassifiedStr = "Correctly Classified: " + std::to_string(corrects) + "/" + std::to_string(total);
    std::string accuracyStr = "Accuracy: " + ossAcc.str() + "%";
    std::string avgLossStr = "Average loss: " + ossLoss.str();

    // Calculate spacing
    int leftPadding = (termWidth - 45 - correctClassifiedStr.length()) / 2;
    int rightPadding = termWidth - (leftPadding + 25 + correctClassifiedStr.length() + accuracyStr.length());

    printf("\n");
    printCentered(title, '*');
    std::cout << "\n" << std::left << correctClassifiedStr;
    std::cout << std::setw(leftPadding + accuracyStr.length()) << std::right << accuracyStr;
    std::cout << std::setw(rightPadding + avgLossStr.length()) << std::right << avgLossStr << "\n" << std::endl;
    printHorizontalLine('*');
}
