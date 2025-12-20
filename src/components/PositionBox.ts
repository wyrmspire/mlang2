/**
 * Position Box Primitive for Lightweight Charts
 * 
 * Draws bounded rectangles for SL/TP zones like TradingView's position tool.
 * Only spans from startTime to endTime, not full-width.
 */

import {
    ISeriesPrimitive,
    SeriesAttachedParameter,
    Time,
    ISeriesPrimitivePaneView,
    ISeriesPrimitivePaneRenderer,
    PrimitiveHoveredItem,
    SeriesPrimitivePaneViewZOrder,
} from 'lightweight-charts';

export interface PositionBoxOptions {
    id: string;
    startTime: Time;
    endTime: Time;
    topPrice: number;
    bottomPrice: number;
    fillColor: string;
    borderColor: string;
    borderWidth?: number;
    label?: string;
    labelColor?: string;
}

class PositionBoxRenderer implements ISeriesPrimitivePaneRenderer {
    private _data: PositionBoxOptions;
    private _x1: number = 0;
    private _x2: number = 0;
    private _y1: number = 0;
    private _y2: number = 0;

    constructor(data: PositionBoxOptions) {
        this._data = data;
    }

    update(x1: number, x2: number, y1: number, y2: number) {
        this._x1 = x1;
        this._x2 = x2;
        this._y1 = y1;
        this._y2 = y2;
    }

    // lightweight-charts v4 uses CanvasRenderingTarget2D which wraps the context
    // We use 'any' to avoid type gymnastics - the runtime API is stable
    draw(target: any): void {
        target.useMediaCoordinateSpace(({ context: ctx }: { context: CanvasRenderingContext2D }) => {
            const width = Math.abs(this._x2 - this._x1);
            const height = Math.abs(this._y2 - this._y1);
            const x = Math.min(this._x1, this._x2);
            const y = Math.min(this._y1, this._y2);

            if (width <= 0 || height <= 0) return;

            // Draw filled rectangle
            ctx.fillStyle = this._data.fillColor;
            ctx.fillRect(x, y, width, height);

            // Draw border
            ctx.strokeStyle = this._data.borderColor;
            ctx.lineWidth = this._data.borderWidth || 1;
            ctx.strokeRect(x, y, width, height);

            // Draw label if provided
            if (this._data.label) {
                ctx.font = 'bold 10px sans-serif';
                ctx.fillStyle = this._data.labelColor || this._data.borderColor;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(this._data.label, x + 4, y + 2);
            }
        });
    }
}

class PositionBoxPaneView implements ISeriesPrimitivePaneView {
    private _source: PositionBox;
    private _renderer: PositionBoxRenderer;

    constructor(source: PositionBox) {
        this._source = source;
        this._renderer = new PositionBoxRenderer(source.options);
    }

    zOrder(): SeriesPrimitivePaneViewZOrder {
        return 'bottom';
    }

    renderer(): ISeriesPrimitivePaneRenderer {
        const series = this._source.series;
        const timeScale = this._source.chart?.timeScale();

        if (!series || !timeScale) {
            this._renderer.update(0, 0, 0, 0);
            return this._renderer;
        }

        const opts = this._source.options;

        // Convert times to x coordinates
        const x1 = timeScale.timeToCoordinate(opts.startTime);
        const x2 = timeScale.timeToCoordinate(opts.endTime);

        // Convert prices to y coordinates
        const y1 = series.priceToCoordinate(opts.topPrice);
        const y2 = series.priceToCoordinate(opts.bottomPrice);

        if (x1 === null || x2 === null || y1 === null || y2 === null) {
            this._renderer.update(0, 0, 0, 0);
            return this._renderer;
        }

        this._renderer.update(x1, x2, y1, y2);
        return this._renderer;
    }
}

export class PositionBox implements ISeriesPrimitive<Time> {
    private _paneViews: PositionBoxPaneView[];
    private _options: PositionBoxOptions;
    private _series: SeriesAttachedParameter<Time> | null = null;

    constructor(options: PositionBoxOptions) {
        this._options = options;
        this._paneViews = [new PositionBoxPaneView(this)];
    }

    get options(): PositionBoxOptions {
        return this._options;
    }

    get series() {
        return this._series?.series ?? null;
    }

    get chart() {
        return this._series?.chart ?? null;
    }

    attached(param: SeriesAttachedParameter<Time>): void {
        this._series = param;
    }

    detached(): void {
        this._series = null;
    }

    paneViews(): readonly ISeriesPrimitivePaneView[] {
        return this._paneViews;
    }

    updateOptions(options: Partial<PositionBoxOptions>): void {
        this._options = { ...this._options, ...options };
    }

    hitTest(): PrimitiveHoveredItem | null {
        return null;
    }
}

// Helper to create SL and TP boxes for a trade
export function createTradePositionBoxes(
    entryPrice: number,
    stopPrice: number,
    tpPrice: number,
    startTime: Time,
    endTime: Time,
    direction: 'LONG' | 'SHORT',
    tradeId: string = 'default'
): { slBox: PositionBox; tpBox: PositionBox; } {

    // SL Zone (red)
    const slBox = new PositionBox({
        id: `sl_${tradeId}`,
        startTime,
        endTime,
        topPrice: Math.max(entryPrice, stopPrice),
        bottomPrice: Math.min(entryPrice, stopPrice),
        fillColor: 'rgba(239, 68, 68, 0.15)',  // red-500 @ 15%
        borderColor: '#ef4444',
        borderWidth: 1,
    });

    // TP Zone (green)
    const tpBox = new PositionBox({
        id: `tp_${tradeId}`,
        startTime,
        endTime,
        topPrice: Math.max(entryPrice, tpPrice),
        bottomPrice: Math.min(entryPrice, tpPrice),
        fillColor: 'rgba(34, 197, 94, 0.15)',  // green-500 @ 15%
        borderColor: '#22c55e',
        borderWidth: 1,
    });

    return { slBox, tpBox };
}
