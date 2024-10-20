async function sendVitalsData(csvFileInput, age, gender, arrivalmode, dispo, symptomsList) {
    // Initialize FormData to send CSV and other fields
    const formData = new FormData();

    // Append the CSV file
    const csvFile = csvFileInput.files[0]; // Get the file from the input
    if (!csvFile) {
        console.error("No file selected.");
        return;
    }
    formData.append("file", csvFile); // Append the CSV file

    // Append other form data
    formData.append("age", age);
    formData.append("gender", gender);
    formData.append("arrivalmode", arrivalmode);
    formData.append("dispo", dispo);
    
    // Create and append symptoms as a JSON string
    const edDataKeys = {
        age: age,
        gender: gender,
        arrivalmode: arrivalmode,
        previousdispo: dispo
    };

    // Add symptoms to the dictionary
    symptomsList.forEach((symptom) => {
        edDataKeys[symptom] = true; // Set to true if the symptom is present
    });

    // Append the edDataKeys as a JSON string
    formData.append("edDataKeys", JSON.stringify(edDataKeys));

    try {
        // Send the POST request
        const response = await fetch("/api/vitals", {
            method: "POST",
            body: formData, // Send formData which contains CSV and other parameters
        });

        if (!response.ok) {
            throw new Error("Error in POST request");
        }

        const data = await response.json();
        console.log("Response from backend:", data);

        // Reset the file input to allow for new file selection
        csvFileInput.value = ""; // Clear the input for re-upload
    } catch (error) {
        console.error("Error:", error);
    }
}
