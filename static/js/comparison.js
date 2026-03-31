const modelViewerComparison1 = document.querySelector("model-viewer#modelViewerComparison1");
const modelViewerComparison2 = document.querySelector("model-viewer#modelViewerComparison2");
const modelViewerComparison3 = document.querySelector("model-viewer#modelViewerComparison3");

// Click an image to select the case

// const CAMERA_ORBIT_Compare = "0deg 75deg 105%";
const CAMERA_ORBIT_Compare = "180deg auto auto";
document.querySelectorAll('#thumbnail-comparison img, #thumbnail-comparison video').forEach(el => {
    el.addEventListener('click', () => {
        const glbSrc = el.getAttribute('data-glb');
        const modelViewer = document.getElementById('modelViewerComparison1');
        modelViewer.setAttribute('src', glbSrc);
        
        // Remove border from all elements
        document.querySelectorAll('#thumbnail-comparison img, #thumbnail-comparison video').forEach(element => {
            element.style.border = 'none';
        });
        
        // Add border to clicked element
        el.style.border = '6px solid #43a3f6';

        const name = el.getAttribute('name');
        console.log('Selected thumbnail name:', name);

        // Store the selected name as an attribute on the thumbnail-comparison container
        document.getElementById('thumbnail-comparison').setAttribute('data-selected-name', name);

        const meshPath1 = `resources/comparison/dvgt/${name}.glb`
        const meshPath2 = `resources/comparison/vggt/${name}.glb`
        const meshPath3 = `resources/comparison/mapanything/${name}.glb`

        modelViewerComparison1.src = meshPath1;
        modelViewerComparison1.cameraOrbit = CAMERA_ORBIT_Compare;
        modelViewerComparison1.resetTurntableRotation(0);
        modelViewerComparison1.jumpCameraToGoal();
        modelViewerComparison2.src = meshPath2;
        modelViewerComparison2.cameraOrbit = CAMERA_ORBIT_Compare;
        modelViewerComparison2.resetTurntableRotation(0);
        modelViewerComparison2.jumpCameraToGoal();
        modelViewerComparison3.src = meshPath3;
        modelViewerComparison3.cameraOrbit = CAMERA_ORBIT_Compare;
        modelViewerComparison3.resetTurntableRotation(0);
        modelViewerComparison3.jumpCameraToGoal();

    });
});
