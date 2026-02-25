package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	fmt.Println("--- Radio-Cortex Log Analyzer (Go) ---")
	
	file, err := os.Open("training.log")
	if err != nil {
		fmt.Println("No training.log found to analyze.")
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineCount := 0
	errorCount := 0

	for scanner.Scan() {
		lineCount++
		if strings.Contains(scanner.Text(), "ERROR") || strings.Contains(scanner.Text(), "Error") {
			errorCount++
		}
	}

	fmt.Printf("Analyzed %d lines.\n", lineCount)
	fmt.Printf("Found %d potential errors/warnings.\n", errorCount)
}
