(function () {
  function disableSubmitButtons(form, submitter) {
    form.querySelectorAll('button[type="submit"]').forEach(function (button) {
      button.disabled = true;
      button.classList.add('disabled');
      button.setAttribute('aria-disabled', 'true');
    });

    if (submitter && submitter.dataset.loadingText) {
      submitter.dataset.originalText = submitter.textContent.trim();
      submitter.textContent = submitter.dataset.loadingText;
    }
  }

  document.querySelectorAll('[data-loading-form]').forEach(function (form) {
    form.addEventListener('submit', function (event) {
      var submitter = event.submitter;

      window.setTimeout(function () {
        disableSubmitButtons(form, submitter);
      }, 0);
    });
  });
})();
