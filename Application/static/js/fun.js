
var map = null;
    var markerArray = [];
    function submit(){

    var val = $("#id_one").val();
    var xhr = new XMLHttpRequest();
    xhr.open('POST', "http://127.0.0.1:80", true);
    var data = new FormData();
    data.append("type", val);
    if(val != "cancer")
      {console.log("Here");
    data.append("img", $("#file_").prop('files')[0]);}
    xhr.onreadystatechange = function(){

      var obj = JSON.parse(this.responseText)[0];
      if(!obj["empty"])
      {
        $("prediction-id").html(obj["top1"]);
        $("prediction-id").html(obj["top2"]);
        $("prediction-id").html(obj["top3"]);
        $("prediction-id").html(obj["top4"]);
        $("prediction-id").html(obj["top5"]);
        if(parseInt(obj["pred_val"]) != 0){
          plotMap(obj["places"]);
          addPlaces(obj["places"]);
        }
        $("#status").html("<b>"+obj["top1"]+"</b>")
        $("#status1").html("<b>"+obj["top2"]+"</b>")
        $("#status2").html("<b>"+obj["top3"]+"</b>")
        $("#status3").html("<b>"+obj["top4"]+"</b>")
        $("#status4").html("<b>"+obj["top5"]+"</b>")
        document.getElementById("demo").innerHTML = "Analysis Report";
        clear();
        $("#result").html("<b>"+obj["pred_val"]+"</b>")

      }
    };
    xhr.send(data);
  }
  function addPlaces(obj){  
    $.each(obj, function(index, obj_){
      console.log(obj_);
      $("#hospital").append("<div><b>"+obj_["name"]+"</b><br>"+obj_["address"]+"</div><br><br><br>");
    });
  }
  function initMap(){
      var  kolkata= {lat: 22.567627, lng: 88.347444};
  // The map, centered at Uluru
    map = new google.maps.Map(
      document.getElementById('map'), {zoom: 10, center: kolkata});
  }
  function clear()
  {
    for(var i=markerArray.length-1; i>=0; i--){
      markerArray[i].setMap(null);
      markerArray.pop();
    }
    $("#hospital").html("");
  }
  function plotMap(places){
    
    $.each(places, function(index,obj){
      if(map != null){
        markerArray.push(new google.maps.Marker({"position": obj["location"], map: map}));
      }
    });
  }
  function changeText(button, text, textToChangeBackTo) {
  buttonId = document.getElementById(button);
  buttonId.textContent = text;
  setTimeout(function() { back(buttonId, textToChangeBackTo); }, 10000);
  function back(button, textToChangeBackTo){ button.textContent = textToChangeBackTo; }
}
function show_hide_cancer(){
  if($("#id_one").val() == "cancer")
    {$("#myDIV").show();
    $("#file_").hide();}
    else{
      $("#myDIV").hide();
    $("#file_").show();
    }
}
