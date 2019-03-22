$('document').ready(function(){
    $('.dataset-select').on('change',function(){
      $.ajax({
      type: "GET",
      url: "/",
      data: {'dataset':  $('.dataset-select').val()}
    });
  });
});
