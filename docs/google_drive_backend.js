/**
 * Google Apps Script for Chelonid-ID Mobile Image Upload
 *
 * Instructions:
 * 1. Go to https://script.google.com/ and create a new project.
 * 2. Paste this entire code into `Code.gs`.
 * 3. Create a folder in your Google Drive to store images (e.g., "Chelonid-ID-Images").
 * 4. Get the Folder ID from the URL of that folder (the part after "folders/").
 * 5. Replace 'YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE' below with that Folder ID.
 * 6. Click Deploy -> New deployment.
 * 7. Select type: Web app.
 * 8. Execute as: "Me" and Who has access: "Anyone" (so the public GitHub Pages site can reach it).
 * 9. Copy the Web app URL and paste it into `docs/index.html` (the `GOOGLE_SCRIPT_URL` variable).
 */

const FOLDER_ID = 'YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE';

function doPost(e) {
  try {
    let data;
    // When using fetch with mode: 'no-cors', the Content-Type header is stripped
    // and the request is sent as text/plain. So we just parse e.postData.contents directly.
    if (e.postData && e.postData.contents) {
        data = JSON.parse(e.postData.contents);
    } else {
        throw new Error("No data received in POST body");
    }

    const filename = data.filename || `upload_${new Date().getTime()}.jpg`;
    const mimeType = data.mimeType || 'image/jpeg';
    const base64Data = data.data;

    // Decode the base64 string back into binary data
    const decodedFile = Utilities.base64Decode(base64Data);

    // Create a blob from the binary data
    const blob = Utilities.newBlob(decodedFile, mimeType, filename);

    // Get the destination folder in Google Drive
    const folder = DriveApp.getFolderById(FOLDER_ID);

    // Create the file in the folder
    const file = folder.createFile(blob);

    // Return a success response
    return ContentService.createTextOutput(JSON.stringify({
      status: 'success',
      fileId: file.getId(),
      fileUrl: file.getUrl()
    })).setMimeType(ContentService.MimeType.JSON);

  } catch (error) {
    // Return an error response
    return ContentService.createTextOutput(JSON.stringify({
      status: 'error',
      message: error.toString()
    })).setMimeType(ContentService.MimeType.JSON);
  }
}

// Needed to handle preflight CORS requests from the browser
function doOptions(e) {
  return ContentService.createTextOutput("")
    .setMimeType(ContentService.MimeType.TEXT);
}
