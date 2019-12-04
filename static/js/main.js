window.onload = function() {
  var btn = document.getElementById('btn');
  btn.addEventListener('click', checkimage, {passive: false} );
  function checkimage(e) {
    var img = document.getElementById('img-input').files;
    if (img.length < 1)
    {
      e.preventDefault();
      alert("すみません！画像がありません！");
    }
  }
}