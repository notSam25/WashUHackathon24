async function sendVitalsData(csvFile, age, gender, arrivalmode, dispo, symptomsList) {
    // Initialize FormData to send CSV and other fields
    const formData = new FormData();
    // formData.append("file", csvFile); // CSV file to send
    // formData.append("age", age); // Add age to formData
    // formData.append("gender", gender);
    // formData.append("dispo", dispo); // Add disposition to formData
    // formData.append("symptoms", JSON.stringify(symptomsList)); // Send symptoms list as a JSON string

    // Initialize edDataKeys with age and previous disposition
    const edDataKeys = {
        age: age,
        gender: gender,
        arrivalmode: arrivalmode,
        previousdispo: dispo,
        // Add more keys as needed for your dictionary
    };

    // Map symptom names to corresponding keys in the dictionary
    symptomsList.forEach((symptom) => {
        if (edDataKeys.hasOwnProperty(symptom)) {
            edDataKeys[symptom] = true; // Set to true if symptom is present
        }
    });

    // Add the updated dictionary to the form data
    formData.append("edDataKeys", JSON.stringify(edDataKeys));

    try {
        const response = await fetch("/api/vitals", {
            method: "POST",
            body: formData, // Send formData which contains CSV and other parameters
        });

        if (!response.ok) {
            throw new Error("Error in POST request");
        }

        const data = await response.json();
        console.log("Response from backend:", data);
    } catch (error) {
        console.error("Error:", error);
    }
}
