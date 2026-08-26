import { useState } from 'react';
import { AnalysisImages } from '../types';
import { Eye, Info, Layers } from 'lucide-react';

interface Props {
  images: AnalysisImages;
  isDiseased?: boolean;
}

export default function ImageAnalysisViewer({ images, isDiseased = true }: Props) {
  const allTabs = [
    {
      id: 'original',
      label: 'Original',
      url: images.original,
      caption: 'Original uploaded plant leaf image.',
    },
    {
      id: 'mask',
      label: 'Disease Mask',
      url: isDiseased ? images.disease_mask : undefined,
      caption: 'Lesion Segmentation: Isolated boundaries of diseased tissue.',
    },
    {
      id: 'gradcam',
      label: 'Grad-CAM',
      url: images.gradcam,
      caption: 'Classifier Attention: Regions influencing the neural network prediction.',
    },
    {
      id: 'overlay',
      label: 'Overlay',
      url: isDiseased ? images.overlay : undefined,
      caption: 'Lesion Overlay: Estimated disease areas highlighted over the leaf.',
    },
  ];

  const availableTabs = allTabs.filter(t => Boolean(t.url));
  const [activeTabId, setActiveTabId] = useState<string>(availableTabs[0]?.id || 'original');

  const activeTab = availableTabs.find(t => t.id === activeTabId) || availableTabs[0];

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col h-full">
      {/* Tabs Header */}
      <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50/80 px-2">
        <div className="flex overflow-x-auto no-scrollbar">
          {availableTabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTabId(tab.id)}
              className={`px-4 py-3 text-xs sm:text-sm font-semibold tracking-wide whitespace-nowrap transition-colors border-b-2 ${
                activeTabId === tab.id
                  ? 'text-emerald-700 border-emerald-600 bg-white shadow-sm'
                  : 'border-transparent text-slate-500 hover:text-slate-800 hover:bg-slate-100/60'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="pr-2 hidden sm:flex items-center text-xs text-slate-400">
          <Layers className="w-3.5 h-3.5 mr-1" />
          <span>Visual Analysis</span>
        </div>
      </div>

      {/* Main Image Viewport */}
      <div className="flex-1 p-4 flex items-center justify-center bg-slate-900/5 min-h-[380px] max-h-[560px] relative overflow-hidden">
        {activeTab?.url ? (
          <img
            src={activeTab.url}
            alt={activeTab.label}
            className="max-w-full max-h-[500px] object-contain rounded-md shadow-sm transition-opacity duration-200"
          />
        ) : (
          <div className="text-slate-400 text-xs flex items-center gap-1.5">
            <Eye className="w-4 h-4" />
            <span>Image stream not available</span>
          </div>
        )}
      </div>

      {/* Caption Bar */}
      {activeTab?.caption && (
        <div className="px-4 py-2.5 bg-slate-50 border-t border-slate-100 flex items-center gap-2 text-xs text-slate-600">
          <Info className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
          <span>{activeTab.caption}</span>
        </div>
      )}
    </div>
  );
}
