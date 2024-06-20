function stopAudio() {
    fetch('/stop_audio', {
        method: 'POST'
    }).then(response => response.json())
      .then(data => {
          if (data.success) {
              console.log("Audio stopped successfully");
          }
      });
}